from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers 

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import save_model

from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt

import os
import cv2
import pickle
from tqdm import tqdm
from absl import app

_INPUT_IMAGE_PRODUCT_METADATA_FILENAME = flags.DEFINE_string(
    'input_image_product_metadata_filename',
    default=None,
    help='The input file name for the Image Product Metadata.',
    required=True)

_INPUT_REQUEST_IMAGE_FILENAME = flags.DEFINE_string(
    'input_request_image_filename',
    default=None,
    help='The input request image file name for similar images.',
    required=True)

class SearchImage():
 # Pretrained Model from tensorflow
 def model_ResNet50_feature_extractor():
  pretrained_model = ResNet50()
  # add a global spatial average pooling layer
  # x = pretrained_model.output
  # x = GlobalAveragePooling2D()(x)
  # x = Dense(1024, activation='relu')(x)
  # # and a logistic layer -- let's say we have 200 classes
  # predictions = Dense(200, activation='softmax')(x)
  model = Model(inputs=pretrained_model.inputs, outputs=pretrained_model.layers[-2].output)
  return model
 
#   def model_inceptionv3_feature_extractor():
#     pretrained_model = InceptionV3()
#     # add a global spatial average pooling layer
#     # x = pretrained_model.output
#     # x = GlobalAveragePooling2D()(x)
#     # x = Dense(1024, activation='relu')(x)
#     # # and a logistic layer -- let's say we have 200 classes
#     # predictions = Dense(200, activation='softmax')(x)
#     model = Model(inputs=pretrained_model.inputs, outputs=pretrained_model.layers[-2].output)
#     return model
  def plot_input_image_features(self,image_request_features):
    plt.figure(figsize=(16,10))
    plt.plot(image_request_features)


  def image_request_features_extract(self, image_path):
    # load an image from file
    image_request = load_img(image_path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image_request = img_to_array(image_request)
    # reshape data for the model
    image_request = image_request.reshape((1, image_request.shape[0], image_request.shape[1], image_request.shape[2]))
    # prepare the image for the VGG model
    image_request = preprocess_input(image_request)
    # get extracted features
    image_request_features = model.predict(image_request)
    print(f'Features Shape of One Input Image: {image_request_features.shape}')
    print(f'Features Dimension of One Input Image: {image_request_features.ndim}')
    print(f'Features of One Input Image: {image_request_features}')
    return image_request_features[0]

  
  def image_database_features(folder_path):
    all_images_features = []
    print("Preapring Feature Vector for all Images")
    for img in tqdm(os.listdir(folder_path)):
      # image_path
      image_path = folder_path + '/' + img
      # load an image from file
      images = load_img(image_path, target_size=(224, 224, 3))
      # convert the image to a numpy array
      images = img_to_array(images)
      # reshape data for the model
      images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
      # prepare the image for the model
      images = preprocess_input(images)
      # get extracted features
      features = model.predict(images)
      # append feature in a list
      all_images_features.append(features[0])
    return all_images_features


  def plot_same_or_similar_images(all_cosine_distance):
    plt.figure(figsize=(16,10))
    all_db_images = os.listdir(all_images_path)
    all_cosine_distance = sorted(all_cosine_distance)
    for idx,i in enumerate(all_cosine_distance[0:8]):
      img = cv2.imread(all_images_path + '/' + all_db_images[i[1]])
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      plt.subplot(3,3,idx+1)
      plt.imshow(img)
      plt.axis("off")
      plt.title(all_db_images[i[1]] + '=' + str(round(i[0],2)))
      plt.tight_layout()
    plt.show()

  def calculate_cosine_distance_with_all_DB_images(query_image_features, all_images_features):
    all_cosine_distance = []
    print("Calculating Cosine Distance with all Images")
    for idx,f in enumerate(all_images_features):
      cos_distance = cosine(query_image_features, f)
      all_cosine_distance.append([cos_distance, idx])
    return all_cosine_distance


  def find_same_or_similar_image(query_image_path):
    query_image_features = image_request_features_extract(query_image_path)
    all_cosine_distance = calculate_cosine_distance_with_all_DB_images(query_image_features, all_images_features)
    plot_same_or_similar_images(all_cosine_distance)
    
  
  model =  model_inceptionv3_feature_extractor()
  all_images_path = _INPUT_IMAGE_PRODUCT_METADATA_FILENAME
  all_images_features = image_database_features(all_images_path)

  def view_image(target_dir):
    # Setup target directory (we'll view images from here)
    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_dir)
    plt.imshow(img)
    plt.title('target_class')
    plt.axis("off");

    print(f"Image shape: {img.shape}") # show the shape of the image

    return img

  request_image_path = _INPUT_REQUEST_IMAGE_FILENAME
  im = view_image( request_image_path )

  query_image_path = request_image_path
  find_same_or_similar_image(query_image_path)

  if __name__ == '__main__':
    main()
