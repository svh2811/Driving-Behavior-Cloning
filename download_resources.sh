# linux simulator (115 MB)
wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip
unzip linux-sim.zip
rm linux-sim.zip

# wget https://files.slack.com/files-pri/T2HQV035L-F3A47CT7Z/download/simulator-linux.zip

cd ./data

# sample training data (317.70 MB)
wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
unzip data.unzip
mv data default_data
rm data.unzip
rm -r ./__MACOSX
