#! /bin/bash
echo "Starting download of the data from Rohban et al. (2017): IDR0033"
cd "$(dirname "$0")"
cd ../..

output_dir = './data/resources/images/rohban/raw'

if [ -d "$output_dir" ]; then
	echo "$output_dir already exist. Exiting to prevent data loss..."
else
	mkdir -p "$output_dir"
	chmod 777 -R "$output_dir"
	cd "$output_dir" || exit
	echo "Downloading ssh key..."
	wget "https://idr.openmicroscopy.org/about/img/aspera/asperaweb_id_dsa.openssh"
	echo "Starting download..."
	mkdir data
	chmod 777 -R data
	
	echo "Downloading imaging data..."
	ascp -TQ -l40m -P 33001 -i "./asperaweb_id_dsa.openssh" idr0033@fasp.ebi.ac.uk:20170214-original/images_illum_corrected ./data
	
	
	echo "Downloading metadata..."
	ascp -TQ -l40m -P 33001 -i "./asperaweb_id_dsa.openssh" idr0033@fasp.ebi.ac.uk:20170214-original/database ./data
	
	echo "Data stored in $output_dir/data"
	echo "Exiting..."
fi
