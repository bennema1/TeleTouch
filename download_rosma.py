#!/usr/bin/env python3
"""
Download ROSMA surgical datasets for cholecystectomy latency compensation training.
Supports both ROSMAG40 and ROSMAT24 datasets from Zenodo.
"""

import requests
import zipfile
import os
from pathlib import Path
import argparse
from tqdm import tqdm


class ROSMADownloader:
    """Download ROSMA cholecystectomy datasets from Zenodo."""

    # Zenodo URLs for ROSMA datasets
    DATASETS = {
        'ROSMAG40': 'https://zenodo.org/records/10719748/files/ROSMAG40.zip',
        'ROSMAT24': 'https://zenodo.org/records/10719714/files/ROSMAT24.zip'
    }

    def __init__(self, download_dir="data/rosma"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_name='ROSMAG40', force_redownload=False):
        """
        Download a specific ROSMA dataset.

        Args:
            dataset_name: 'ROSMAG40' or 'ROSMAT24'
            force_redownload: If True, re-download even if already exists
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}")

        dataset_path = self.download_dir / dataset_name
        zip_path = self.download_dir / f"{dataset_name}.zip"

        # Check if already downloaded
        if dataset_path.exists() and not force_redownload:
            print(f"[OK] {dataset_name} already downloaded at {dataset_path}")
            return dataset_path

        # Download the dataset
        url = self.DATASETS[dataset_name]
        print(f"Downloading {dataset_name} from Zenodo...")
        print(f"URL: {url}")

        try:
            # Stream download with progress bar
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f, tqdm(
                desc=f"Downloading {dataset_name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Extract the zip file
            print(f"📦 Extracting {dataset_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.download_dir)

            # Clean up zip file
            os.remove(zip_path)

            print(f"✅ {dataset_name} downloaded and extracted successfully!")
            print(f"📁 Location: {dataset_path}")

            # Verify contents
            self._verify_dataset(dataset_path, dataset_name)

            return dataset_path

        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            if zip_path.exists():
                os.remove(zip_path)
            raise

    def _verify_dataset(self, dataset_path, dataset_name):
        """Verify the downloaded dataset has expected contents."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        # Check for expected file types
        json_files = list(dataset_path.glob("*.json"))
        csv_files = list(dataset_path.glob("*kinematic*.csv"))

        print(f"📊 Dataset verification:")
        print(f"   JSON annotation files: {len(json_files)}")
        print(f"   CSV kinematic files: {len(csv_files)}")

        if len(json_files) == 0 or len(csv_files) == 0:
            print("⚠️ Warning: Expected files not found. Dataset may be incomplete.")

        # Print some examples
        if json_files:
            print(f"   Sample JSON files: {json_files[:3]}")
        if csv_files:
            print(f"   Sample CSV files: {csv_files[:3]}")

    def download_all_datasets(self, force_redownload=False):
        """Download both ROSMA datasets."""
        print("🚀 Downloading all ROSMA cholecystectomy datasets...")
        print("=" * 60)

        downloaded_paths = []
        for dataset_name in self.DATASETS.keys():
            try:
                path = self.download_dataset(dataset_name, force_redownload)
                downloaded_paths.append(path)
                print()
            except Exception as e:
                print(f"❌ Failed to download {dataset_name}: {e}")
                print()

        print("📋 Download Summary:")
        for path in downloaded_paths:
            print(f"   ✅ {path.name}: {path}")

        print(f"\n💾 Total datasets downloaded: {len(downloaded_paths)}/{len(self.DATASETS)}")
        print(f"📁 All data saved in: {self.download_dir}")

        return downloaded_paths

    def get_dataset_info(self):
        """Get information about available datasets."""
        info = {
            'ROSMAG40': {
                'description': 'ROSMA Gallbladder dataset with 40 cholecystectomy procedures',
                'size': '~4GB',
                'procedures': 'Laparoscopic cholecystectomy (gallbladder removal)',
                'features': 'Instrument kinematics, video annotations, surgical phases'
            },
            'ROSMAT24': {
                'description': 'ROSMA Training dataset with 24 cholecystectomy procedures',
                'size': '~2.5GB',
                'procedures': 'Laparoscopic cholecystectomy training scenarios',
                'features': 'Instrument trajectories, kinematic data, annotation labels'
            }
        }
        return info


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description="Download ROSMA cholecystectomy datasets")
    parser.add_argument('--dataset', type=str, default='ROSMAG40',
                       choices=['ROSMAG40', 'ROSMAT24', 'all'],
                       help='Which dataset to download (default: ROSMAG40)')
    parser.add_argument('--download-dir', type=str, default='data/rosma',
                       help='Directory to save datasets (default: data/rosma)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if already exists')
    parser.add_argument('--info', action='store_true',
                       help='Show dataset information and exit')

    args = parser.parse_args()

    downloader = ROSMADownloader(args.download_dir)

    if args.info:
        info = downloader.get_dataset_info()
        print("📚 ROSMA Cholecystectomy Datasets:")
        print("=" * 50)
        for name, details in info.items():
            print(f"\n{name}:")
            for key, value in details.items():
                print(f"  {key.capitalize()}: {value}")
        return

    if args.dataset == 'all':
        downloader.download_all_datasets(force_redownload=args.force)
    else:
        downloader.download_dataset(args.dataset, force_redownload=args.force)


if __name__ == "__main__":
    main()