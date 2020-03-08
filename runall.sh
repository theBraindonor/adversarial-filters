# Please note that all Imagenet validation set images and labels need to be 
# downloaded locally into the data folder before this script can be run.  So long
# as the first script that processes the Imagenet labels runs correctly, then
# everything should proceed correctly.
sample=100
# Preproces the data
python -m preproces.create_imagenet_labels
python -m preprocess.create_vgg16_dataset -n $sample
python -m preprocess.create_vgg19_dataset -n $sample
python -m preprocess.create_densenet201_dataset -n $sample
python -m preprocess.create_resnet152v2_dataset -n $sample
# Run the experiments for the network
python -m experiment.color_blindness_filter -n vgg16 -s $sample
python -m experiment.color_blindness_filter -n vgg19 -s $sample
python -m experiment.color_blindness_filter -n densenet -s $sample
python -m experiment.color_blindness_filter -n resnet -s $sample
python -m experiment.daltonization_filter -n vgg16 -s $sample
python -m experiment.daltonization_filter -n vgg19 -s $sample
python -m experiment.daltonization_filter -n densenet -s $sample
python -m experiment.daltonization_filter -n resnet -s $sample
python -m experiment.daltonization_lab_enhance_filter -n vgg16 -s $sample
python -m experiment.daltonization_lab_enhance_filter -n vgg19 -s $sample
python -m experiment.daltonization_lab_enhance_filter -n densenet -s $sample
python -m experiment.daltonization_lab_enhance_filter -n resnet -s $sample
python -m experiment.fourier_ellipsoid_filter -n vgg16 -s $sample
python -m experiment.fourier_ellipsoid_filter -n vgg19 -s $sample
python -m experiment.fourier_ellipsoid_filter -n densenet -s $sample
python -m experiment.fourier_ellipsoid_filter -n resnet -s $sample
