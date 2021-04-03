# some training parameters
EPOCHS = 4
BATCH_SIZE = 896
NUM_CLASSES = 10
image_height = 32
image_width = 32
channels = 3
save_model_dir = "saved_model/model"
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"

# choose a network
# model = "resnet18"
model = "resnet34"
#model = "resnet50"
# model = "resnet101"
# model = "resnet152"
