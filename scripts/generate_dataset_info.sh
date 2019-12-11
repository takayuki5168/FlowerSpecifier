cd $(dirname $0)/../dataset

info_file_name="info.txt"


# move to train directory
cd train
train_dirs="$(ls -d */)"
echo $train_dirs

# init info.txt
rm ${info_file_name} -f
touch ${info_file_name}


# write info.txt
idx=0
for train_dir in $train_dirs; do
    files="${train_dir}/*"
    for file in $files; do
     	echo "${file}" "$idx" >> "${info_file_name}"
    done
    idx=$((idx+1))
done


# move to validation directory
cd ../validation
validation_dirs="$(ls -d */)"

# init info.txt
rm ${info_file_name} -f
touch ${info_file_name}

# write info.txt
idx=0
for validation_dir in $validation_dirs; do
    files="${validation_dir}/*"
    for file in $files; do
     	echo "${file}" "$idx" >> "${info_file_name}"
    done
    idx=$((idx+1))
done
