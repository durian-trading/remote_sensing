#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

### The first model in the report
def model1(input_size = (256,256,1)):
    
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    merge1 = concatenate([conv1,inputs], axis = 3)
    
    dense2 = Dense(128, activation = 'relu')(merge1)
    
    conv3 = Conv2D(1,1,activation='sigmoid')(dense2)

    model = Model(input = inputs, output = conv3)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['mse'])

    return model

###Extract an image from a catalog entry
def get_crop_from_aoi(output_path, aoi, catalog_entry, band):
    metadata = catalog_entry
    if not metadata.urls['gcloud']:
        metadata.build_gs_links()
    inpath = metadata.urls['gcloud'][band]
    utm_zone = int(metadata.utm_zone) if 'utm_zone' in metadata else None
    ulx, uly, lrx, lry, utm_zone, lat_band = tsd.utils.utm_bbx(aoi, utm_zone=utm_zone,  r=60)
    tsd.utils.crop_with_gdal_translate(output_path, inpath, ulx, uly, lrx, lry, utm_zone, lat_band)

### Uses the picture's max and min value to adapt its contrast
def simple_equalization_8bit(im, percentiles=5):
    res = np.minimum(im,np.percentile(im,100-percentiles))
    res = np.maximum(res,np.percentile(res,percentiles))
    M,m = np.max(res),np.min(res)
    return np.round((res-m)*255/(M-m))

### Fetches the channels for a catalog_entry
def get_sentinel2_channels(basefilename, aoi, catalog_entry, all_bands = True):
    color_bands = ['B02','B03','B04']
    bands = color_bands
    if all_bands:
        bands = ['B0' + str(i+2) for i in range(7)]
    for b in bands:
        get_crop_from_aoi('{}_{}.tif'.format(basefilename, b), aoi, catalog_entry, b)
    out = []
    for b in bands:
        im = utils.readGTIFF('{}_{}.tif'.format(basefilename, b))
        out.append(im)
    min_dim = min([x.shape[0] for x in out])
    sub_sample_fact = [x.shape[0]//min_dim for x in out]
    for i,x in enumerate(out):
        fact = sub_sample_fact[i]
        out[i] = x[::fact,::fact]
    im = np.squeeze(out,axis=(3)).transpose(1,2,0)
    return np.array(im)

### Computes the NDVI of a catalog_entry
def NDVI_from_aoi(basefilename, aoi, catalog_entry):
    bands = ['B04','B08']
    for b in bands:
        get_crop_from_aoi('{}_{}.tif'.format(basefilename, b), aoi, catalog_entry, b)
    red = utils.readGTIFF('{}_{}.tif'.format(basefilename,bands[0]))
    nir = utils.readGTIFF('{}_{}.tif'.format(basefilename,bands[1]))
    NDVI = (nir - red) / (nir + red)
    return np.maximum(0.,NDVI)

### NDVI of a (:,:,7) image
def NDVI(im):
    red, nir = im[:,:,2], im[:,:,6]
    return np.maximum(0.,(nir-red)/np.where((nir+red) == 0, 1, nir+red)).astype('double')

### NDWI of a (:,:,7) image
def NDWI(im):
    green, nir = im[:,:,1], im[:,:,6]
    return np.maximum(0.,(green-nir)/np.where((green+nir)==0,1,green+nir)).astype('double')

class Timeout(Exception): 
    pass 

### The main extraction function. Fetches the number of images for a given aoi. If one image takes too long,
### it is skipped (prevents some bugs)
def generate_database(basename, aoi, size=(64,64), max_cloud_cover = 10, nb_images = 1,timeout=800):
    def timeout_handler(signum, frame):
        raise Timeout()
    train_set = []
    boa_catalog = tsd.get_sentinel2.search(aoi, product_type = 'L2A', api='scihub')
    toa_catalog = tsd.get_sentinel2.search(aoi, product_type = 'L1C', api='scihub')
    cloud_catalog = tsd.get_sentinel2.search(aoi)
    list_boa, list_toa = get_pair_cloud(boa_catalog,toa_catalog,cloud_catalog,max_cloud_cover=max_cloud_cover)
    boa,toa = [],[]
    j = 0 
    for i in range(len(list_boa)):
        print(f'fetching image {i}')
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            boa += get_crops(get_sentinel2_channels(basename,aoi,list_boa[i],all_bands=True),size)
            toa += get_crops(get_sentinel2_channels(basename,aoi,list_toa[i],all_bands=True),size)
            j += 1
        except Timeout:
            print(f'picture {i} failed')
        finally:
            signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)
        if j == nb_images:
            break
    print(f'got {j} images')
    train_set += [{'feat_boa':boa[i],'feat_toa':toa[i],'NDVI':NDVI(boa[i]),'NDWI':NDWI(boa[i])} for i in range(len(boa))]
    return train_set


### Returns a pair of boa / toa with low cloud cover
def get_pair_cloud(boa_catalog,toa_catalog,cloud_catalog,max_cloud_cover=10, max_delta = 10):
    cloud_catalog = sorted(cloud_catalog, key = lambda x: x['cloud_cover'])
    list_boa, list_toa = [],[]
    j = 0
    for date in [x['date'] for x in cloud_catalog if x['cloud_cover'] < max_cloud_cover]:
        return_boa = None
        return_toa = None
        for boa in boa_catalog:
            if abs(boa['date'] - date).total_seconds() < max_delta:
                return_boa = boa
        for toa in toa_catalog:
            if abs(toa['date'] - date).total_seconds() < max_delta:
                return_toa = toa
        if not (return_boa == None or return_toa == None):
            list_boa.append(return_boa)
            list_toa.append(return_toa)
    return list_boa,list_toa

### Generate a rectangular aoi of fixed size from a circle
def generate_aoi(raw_aoi, x, y):
    a,b = raw_aoi['geometry']['coordinates']
    return {'type': 'Polygon',
           'coordinates': [[[a-x,b-y],[a-x,b+y],[a+x,b+y],[a+x,b-y],[a-x,b-y]]]}

### Splits the image in distinct images of size = size
def get_crops(image,size):
    x,y = size[0],size[1]
    n_x,n_y = image.shape[0]//x,image.shape[1]//y
    return [image[i*x:(i+1)*x,j*y:(j+1)*y,:] for i,j in product(range(n_x),range(n_y))]

### Takes an image of size = size at the center of the image
def center_image(image,size):
    x,y = size[0],size[1]
    x_mid,y_mid = image.shape[0]//2,image.shape[1]//2
    return image[x_mid-round(x/2):x_mid+ceil(x/2),y_mid-round(y/2):y_mid+ceil(y/2),:]

### Supposed to fetch a time series using the tsd API
### IT DOES NOT WORK IN PRACTICE (image often not found)
def time_series(aoi):
    return tsd.get_sentinel2.get_time_series(aoi, bands=['B0' + str(i+2) for i in range(7)], 
            api='scihub', cloud_masks = True, output_dir = 'tmp')
    for b in bands:
        get_crop_from_aoi('{}_{}.tif'.format(basefilename, b), aoi, catalog_entry, b)
    out = []
    for b in bands:
        im = utils.readGTIFF('{}_{}.tif'.format(basefilename, b))
        out.append(im)
    min_dim = min([x.shape[0] for x in out])
    sub_sample_fact = [x.shape[0]//min_dim for x in out]
    for i,x in enumerate(out):
        fact = sub_sample_fact[i]
        out[i] = x[::fact,::fact]
    im = np.squeeze(out,axis=(3)).transpose(1,2,0)
    return np.array(im)

### Applies a model to a list of images, by splitting the images in smaller images
def apply_model(images,model):
    x,y = model.input_shape[1:3]
    n_image,x_image,y_image = images.shape[0],images.shape[1],images.shape[2]
    n_x,n_y = x_image//x,y_image//y
    index_image = np.zeros((n_image,x_image - x_image%x,y_image - y_image%y))
    for i,j in product(range(n_x),range(n_y)):
        index_image[:,i*x:(i+1)*x,j*y:(j+1)*y]=model.predict(images[:,i*x:(i+1)*x,j*y:(j+1)*y,:])[:,:,:,0]
    return index_image

### Computes the model's score and compares it to the ToA NDVI
def model_score(model, features, target, index_func):
    prediction = apply_model(features,model)[:,:,:,np.newaxis]
    return ((target - prediction)**2).mean(), ((target - np.array([index_func(x)[:,:,np.newaxis] for x in features]))**2).mean()

### Display a 2D image after equalization
display_color = lambda x: vistools.display_image(simple_equalization_8bit(x[:,:,0:3]))

### Codes used for the second model
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    print(f'input shape {input_img.shape}')
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)
    print(f'p1 shape : {p1.shape}')
    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)
    print(f'p2 shape : {p2.shape}')
    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)
    print(f'p3 shape : {p3.shape}')
    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    print(f'p4 shape : {p4.shape}')
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    print(f'c5 shape : {c5.shape}')
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    print(f'u6 shape : {u6.shape} et c4.shape {c4.shape}')
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    print(f'c6 shape {c6.shape}')
    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    print(f'u7 shape : {u7.shape} et c3.shape {c3.shape}')
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    print(f'u8 shape : {u8.shape} et c2.shape {c2.shape}')
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    print(f'c9.shape {c9.shape}')
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    print(f'outputs shape {outputs.shape}')
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def get_small_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    print(f'input shape {input_img.shape}')
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)
    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)
    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    # expansive path
    u4 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c3)
    u4 = concatenate([u4, c2])
    u4 = Dropout(dropout)(u4)
    c4 = conv2d_block(u4, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    
    u5 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c4)
    u5 = concatenate([u5, c1])
    u5 = Dropout(dropout)(u5)
    c5 = conv2d_block(u5, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c5)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model

