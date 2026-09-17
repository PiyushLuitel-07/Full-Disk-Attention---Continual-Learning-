import csv
import datetime
import os
import re
import time
from pathlib import Path

import cv2
import requests


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

START_DATE = '2010-12-01 00:00:00'
END_DATE = '2018-12-31 23:59:59'

CADENCE_MINUTES = 60
MAXIMUM_TIME_DIFFERENCE_MINUTES = 12
MAXIMUM_RETRIES = 5

SOURCE_ID = 19

# This file is inside repo/download_mag/, so parent.parent is repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

JP2_DIRECTORY = PROJECT_ROOT / 'downloaded_data' / 'hmi_compressed'
JPG_DIRECTORY = PROJECT_ROOT / 'downloaded_data' / 'hmi_jpgs'
LOG_FILE = PROJECT_ROOT / 'downloaded_data' / 'download_log.csv'


def write_log(
    requested_time,
    actual_time,
    status,
    filename,
    attempts,
    message
):
    """Record the result of one requested timestamp."""

    create_header = not LOG_FILE.exists()

    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if create_header:
            writer.writerow([
                'requested_time',
                'actual_time',
                'status',
                'filename',
                'attempts',
                'message'
            ])

        writer.writerow([
            requested_time,
            actual_time,
            status,
            filename,
            attempts,
            message
        ])


def download_from_helioviewer():
    """
    Download HMI magnetograms from Helioviewer at a 12-minute cadence.

    Existing files are skipped, allowing the script to resume after
    interruption.
    """

    JP2_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    start = datetime.datetime.strptime(
        START_DATE,
        '%Y-%m-%d %H:%M:%S'
    )
    stop = datetime.datetime.strptime(
        END_DATE,
        '%Y-%m-%d %H:%M:%S'
    )

    current_time = start
    session = requests.Session()

    downloaded_count = 0
    existing_count = 0
    unavailable_count = 0
    failed_count = 0

    while current_time <= stop:
        requested_time = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        output_directory = (
            JP2_DIRECTORY
            / str(current_time.year)
            / f'{current_time.month:02d}'
            / f'{current_time.day:02d}'
        )
        output_directory.mkdir(parents=True, exist_ok=True)

        filename = (
            f'HMI.m{current_time.year}.'
            f'{current_time.month:02d}.'
            f'{current_time.day:02d}_'
            f'{current_time.hour:02d}.'
            f'{current_time.minute:02d}.'
            f'{current_time.second:02d}.jp2'
        )

        final_file = output_directory / filename
        temporary_file = Path(str(final_file) + '.part')

        # Resume support: do not download completed files again.
        if final_file.exists() and final_file.stat().st_size > 0:
            existing_count += 1
            current_time += datetime.timedelta(
                minutes=CADENCE_MINUTES
            )
            continue

        status = 'failed'
        actual_time_text = ''
        error_message = ''
        attempts_used = 0

        for attempt in range(1, MAXIMUM_RETRIES + 1):
            attempts_used = attempt

            try:
                # Ask Helioviewer which image it considers closest.
                metadata_response = session.get(
                    'https://api.helioviewer.org/v2/getJP2Image/',
                    params={
                        'date': requested_time,
                        'sourceId': SOURCE_ID,
                        'jpip': 'true',
                        'json': 'true'
                    },
                    timeout=(15, 60)
                )
                metadata_response.raise_for_status()

                uri = metadata_response.json().get('uri', '')

                timestamp_match = re.search(
                    r'(\d{4}_\d{2}_\d{2}__'
                    r'\d{2}_\d{2}_\d{2})',
                    uri
                )

                if timestamp_match is None:
                    raise ValueError(
                        'No image timestamp found in Helioviewer response'
                    )

                actual_time = datetime.datetime.strptime(
                    timestamp_match.group(1),
                    '%Y_%m_%d__%H_%M_%S'
                )
                actual_time_text = actual_time.strftime(
                    '%Y-%m-%d %H:%M:%S'
                )

                time_difference = abs(actual_time - current_time)

                # Reject a distant image returned as the nearest image.
                if time_difference > datetime.timedelta(
                    minutes=MAXIMUM_TIME_DIFFERENCE_MINUTES
                ):
                    status = 'unavailable'
                    error_message = (
                        f'Nearest image is {time_difference} away'
                    )
                    unavailable_count += 1
                    break

                # Download the actual JP2 data.
                with session.get(
                    'https://api.helioviewer.org/v2/getJP2Image/',
                    params={
                        'date': requested_time,
                        'sourceId': SOURCE_ID
                    },
                    stream=True,
                    timeout=(15, 120)
                ) as image_response:
                    image_response.raise_for_status()

                    content_type = image_response.headers.get(
                        'Content-Type',
                        ''
                    ).lower()

                    if (
                        'text/html' in content_type
                        or 'application/json' in content_type
                    ):
                        raise ValueError(
                            f'Unexpected response type: {content_type}'
                        )

                    with open(temporary_file, 'wb') as output_file:
                        for chunk in image_response.iter_content(
                            chunk_size=1024 * 1024
                        ):
                            if chunk:
                                output_file.write(chunk)

                if (
                    not temporary_file.exists()
                    or temporary_file.stat().st_size == 0
                ):
                    raise ValueError('Downloaded file is empty')

                # A final JP2 filename is used only after completion.
                temporary_file.replace(final_file)

                status = 'downloaded'
                error_message = f'{final_file.stat().st_size} bytes'
                downloaded_count += 1
                break

            except (
                requests.RequestException,
                ValueError,
                OSError
            ) as error:
                error_message = str(error)

                if temporary_file.exists():
                    try:
                        temporary_file.unlink()
                    except OSError:
                        pass

                if attempt < MAXIMUM_RETRIES:
                    wait_seconds = min(60, 2 ** attempt)

                    print(
                        f'{requested_time}: attempt {attempt} failed: '
                        f'{error}. Retrying in {wait_seconds} seconds.',
                        flush=True
                    )
                    time.sleep(wait_seconds)
                else:
                    status = 'failed'
                    failed_count += 1

        write_log(
            requested_time,
            actual_time_text,
            status,
            str(final_file),
            attempts_used,
            error_message
        )

        print(
            f'{requested_time} | {status} | '
            f'downloaded={downloaded_count}, '
            f'existing={existing_count}, '
            f'unavailable={unavailable_count}, '
            f'failed={failed_count}',
            flush=True
        )

        current_time += datetime.timedelta(
            minutes=CADENCE_MINUTES
        )

    print('JP2 download stage completed.', flush=True)


def jp2_to_jpg_conversion(resize=True, width=512, height=512):
    """
    Convert downloaded JP2 files to JPG.

    Existing JPG files are skipped, making conversion restartable.
    JP2 files are preserved.
    """

    JPG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    existing_count = 0
    failed_count = 0

    for source_file in JP2_DIRECTORY.rglob('*.jp2'):
        relative_path = source_file.relative_to(JP2_DIRECTORY)
        destination_file = (
            JPG_DIRECTORY / relative_path
        ).with_suffix('.jpg')

        destination_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if (
            destination_file.exists()
            and destination_file.stat().st_size > 0
        ):
            existing_count += 1
            continue

        try:
            image = cv2.imread(
                str(source_file),
                cv2.IMREAD_UNCHANGED
            )

            if image is None:
                raise ValueError('OpenCV could not read the JP2 file')

            if resize:
                image = cv2.resize(
                    image,
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )

            success = cv2.imwrite(
                str(destination_file),
                image
            )

            if not success:
                raise ValueError('OpenCV could not write the JPG file')

            converted_count += 1

            print(
                f'{source_file} -> {destination_file}',
                flush=True
            )

        except (ValueError, OSError, cv2.error) as error:
            failed_count += 1
            print(
                f'Conversion failed for {source_file}: {error}',
                flush=True
            )

    print(
        f'Conversion completed: converted={converted_count}, '
        f'existing={existing_count}, failed={failed_count}',
        flush=True
    )


if __name__ == '__main__':
    download_from_helioviewer()
    jp2_to_jpg_conversion(
        resize=True,
        width=512,
        height=512
    )
