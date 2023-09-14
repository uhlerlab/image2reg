cd data/resources/images/rohban/raw/data/database

echo "create database TargetAccelerator" | mysql -u root -p
mysql -u root -p TargetAccelerator < TargetAccelerator.sql

cd ../../../../../../..

if ! [ -d "data/resources/images/rohban/metadata" ]; then
	mkdir data/resources/images/rohban/metadata
fi
chmod 777 -R data/resources/images/rohban/metadata

mysql -u root -p TargetAccelerator < scripts/data/extract_metadata.sql | scripts/data/tab2csv > data/resources/images/rohban/metadata/metadata_images.csv

mysql -u root -p TargetAccelerator < scripts/data/extract_profile_data.sql | scripts/data/tab2csv > data/resources/images/rohban/metadata/nuclei_morph_profiles.csv


mv data/resources/images/rohban/raw/data/images_illum_corrected data/resources/images/rohban/illum_corrected
