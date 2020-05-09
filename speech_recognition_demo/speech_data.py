"""
Utilities for downloading and providing data from 
openslr.org, libriSpeech, Pannous, Gutenberg, WMT, tokenizing, vocabularies.

NOTE: see https://github.com/pannous/caffe-speech-recognition for some data sources
"""
# standard
import os
import random
import re
import sys
import wave
# third-party
from enum import Enum
import librosa
import matplotlib
import numpy
from six.moves import urllib
import skimage.io

SOURCE_URL = 'http://pannous.net/files/'
DATA_DIR = 'data/'
pcm_path = "data/spoken_numbers_pcm/"
wav_path = "data/spoken_numbers_wav/"
path = pcm_path
CHUNK = 4096
test_fraction=0.1

class DataSet:
    """
    DOCSTRING
    """
    def __init__(self, images, labels, fake_data=False, one_hot=False, load=False):
        """
        Construct a DataSet. one_hot arg is used only if fake_data is true.
        """
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            num = len(images)
            assert num == len(labels), ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            print("len(images) %d" % num)
            self._num_examples = num
        self.cache={}
        self._image_names = numpy.array(images)
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._images=[]
        if load:
            self._images=self.load(self._image_names)

    @property
    def images(self):
        return self._images

    @property
    def image_names(self):
        return self._image_names

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def load(self,image_names):
        """
        NOTE: only apply to a subset of all images at one time
        """
        print("loading %d images"%len(image_names))
        return list(map(self.load_image,image_names))

    def load_image(self,image_name):
        """
        DOCSTRING
        """
        if image_name in self.cache:
            return self.cache[image_name]
        else:
            image = skimage.io.imread(DATA_DIR+ image_name).astype(numpy.float32)
            self.cache[image_name]=image
            return image

    def next_batch(self, batch_size, fake_data=False):
        """
        Return the next `batch_size` examples from this data set.
        """
        if fake_data:
            fake_image = [1] * width * height
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._image_names = self._image_names[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.load(self._image_names[start:end]), self._labels[start:end]

class Source:
    """
    DOCSTRING
    """
    DIGIT_WAVES = 'spoken_numbers_pcm.tar'
    DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'
    NUMBER_WAVES = 'spoken_numbers_wav.tar'
    NUMBER_IMAGES = 'spoken_numbers.tar'
    WORD_SPECTROS = 'https://dl.dropboxusercontent.com/u/23615316/spoken_words.tar'
    TEST_INDEX = 'test_index.txt'
    TRAIN_INDEX = 'train_index.txt'

class Target(Enum):
    """
    DOCSTRING
    """
    digits=1
    speaker=2
    words_per_minute=3
    word_phonemes=4
    word=5
    sentence=6
    sentiment=7
    first_letter=8

def dense_to_one_hot(labels_dense, num_classes=10):
    """
    Convert class labels from scalars to one-hot vectors.
    """
    return numpy.eye(num_classes)[labels_dense]

def dense_to_one_hot(batch, batch_size, num_labels):
    """
    DOCSTRING
    """
    sparse_labels = tf.reshape(batch, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    concatenated = tf.concat(1, [indices, sparse_labels])
    concat = tf.concat(0, [[batch_size], [num_labels]])
    output_shape = tf.reshape(concat, [2])
    sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
    return tf.reshape(sparse_to_dense, [batch_size, num_labels])

def dense_to_some_hot(labels_dense, num_classes=140):
    """
    Convert class labels from int vectors to many-hot vectors
    """
    raise "TODO: dense_to_some_hot"

def extract_images(names_file,train):
    """
    DOCSTRING
    """
    image_files=[]
    for line in open(names_file).readlines():
        image_file,image_label = line.split("\t")
        image_files.append(image_file)
    return image_files

def extract_labels(names_file,train, one_hot):
    """
    DOCSTRING
    """
    labels=[]
    for line in open(names_file).readlines():
        image_file,image_label = line.split("\t")
        labels.append(image_label)
    if one_hot:
        return dense_to_one_hot(labels)
    return labels

def get_speakers(path=pcm_path):
    """
    DOCSTRING
    """
    files = os.listdir(path)
    def nobad(file):
        return "_" in file and not "." in file.split("_")[1]
    speakers=list(set(map(speaker,filter(nobad,files))))
    print(len(speakers)," speakers: ",speakers)
    return speakers

def load_wav_file(name):
    """
    DOCSTRING
    """
    f = wave.open(name, "rb")
    chunk = []
    data0 = f.readframes(CHUNK)
    while data0:
        data = numpy.fromstring(data0, dtype='uint8')
        data = (data + 128) / 255.0
        chunk.extend(data)
        data0 = f.readframes(CHUNK)
    chunk = chunk[0:CHUNK * 2] 
    chunk.extend(numpy.zeros(CHUNK * 2 - len(chunk)))
    return chunk

def maybe_download(file, work_directory):
    """
    Download the data from Pannous's website, unless it's already here.
    """
    print("Looking for data %s in %s"%(file,work_directory))
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, re.sub('.*\/','',file))
    if not os.path.exists(filepath):
        if not file.startswith("http"): 
            url_filename = SOURCE_URL + file
        else: 
            url_filename=file
        print('Downloading from %s to %s' % (url_filename, filepath))
        filepath, _ = urllib.request.urlretrieve(url_filename, filepath,progresshook)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', file, statinfo.st_size, 'bytes.')
        # os.system('ln -s '+work_directory)
    if os.path.exists(filepath):
        print('Extracting %s to %s' % ( filepath, work_directory))
        os.system('tar xf %s -C %s' % ( filepath, work_directory))
        print('Data ready!')
    return filepath.replace(".tar","")

def mfcc_batch_generator(batch_size=10, source=Source.DIGIT_WAVES, target=Target.digits):
    """
    DOCSTRING
    """
    maybe_download(source, DATA_DIR)
    if target == Target.speaker: 
        speakers = get_speakers()
    batch_features = []
    labels = []
    files = os.listdir(path)
    while True:
        print("loaded batch of %d files" % len(files))
        random.shuffle(files)
        for wav in files:
            if not wav.endswith(".wav"):
                continue
            wave, sr = librosa.load(path+wav, mono=True)
            if target==Target.speaker: 
                label=one_hot_from_item(speaker(wav), speakers)
            elif target==Target.digits:
                label=dense_to_one_hot(int(wav[0]),10)
            elif target==Target.first_letter:
                label=dense_to_one_hot((ord(wav[0]) - 48) % 32, 32)
            else: 
                raise Exception("todo : labels for Target!")
            labels.append(label)
            mfcc = librosa.feature.mfcc(wave, sr)
            mfcc=numpy.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
            batch_features.append(numpy.array(mfcc))
            if len(batch_features) >= batch_size:
                yield batch_features, labels
                batch_features = []
                labels = []

def one_hot_from_item(item, items):
    """
    DOCSTRING
    """
    x = [0]*len(items)
    i = items.index(item)
    x[i] = 1
    return x

def one_hot_to_item(hot, items):
    """
    DOCSTRING
    """
    i=numpy.argmax(hot)
    item=items[i]
    return item

def progresshook(blocknum, blocksize, totalsize):
    """
    DOCSTRING
    """
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:
            sys.stderr.write("\n")
    else:
        sys.stderr.write("read %d\n" % (readsofar,))

def read_data_sets(train_dir, source_data=Source.NUMBER_IMAGES, fake_data=False, one_hot=True):
    """
    DOCSTRING
    """
    class DataSets:
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True, one_hot=one_hot)
        data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
        data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
        return data_sets
    VALIDATION_SIZE = 2000
    local_file = maybe_download(source_data, train_dir)
    train_images = extract_images(TRAIN_INDEX,train=True)
    train_labels = extract_labels(TRAIN_INDEX,train=True, one_hot=one_hot)
    test_images = extract_images(TEST_INDEX,train=False)
    test_labels = extract_labels(TEST_INDEX,train=False, one_hot=one_hot)
    data_sets.train = DataSet(train_images, train_labels , load=False)
    data_sets.test = DataSet(test_images, test_labels, load=True)
    return data_sets

def speaker(file):
    """
    DOCSTRING
    """
    return file.split("_")[1]

def spectro_batch(batch_size=10):
    """
    DOCSTRING
    """
    return spectro_batch_generator(batch_size)

def spectro_batch_generator(batch_size=10,
                            width=64,
                            source_data=Source.DIGIT_SPECTROS,
                            target=Target.digits):
    """
    DOCSTRING
    """
    path=maybe_download(source_data, DATA_DIR)
    path=path.replace("_spectros", "")
    height = width
    batch = []
    labels = []
    speakers=get_speakers(path)
    if target==Target.digits:
        num_classes = 10
    if target==Target.first_letter:
        num_classes = 32
    files = os.listdir(path)
    print("Got %d source data files from %s"%(len(files),path))
    while True:
        random.shuffle(files)
        for image_name in files:
            if not "_" in image_name: 
                continue
            image = skimage.io.imread(path + "/" + image_name).astype(numpy.float32)
            data = image / 255.0
            data = data.reshape([width * height])
            batch.append(list(data))
            classe = (ord(image_name[0]) - 48) % 32
            labels.append(dense_to_one_hot(classe, num_classes))
            if len(batch) >= batch_size:
                yield batch, labels
                batch = []
                labels = []

def wave_batch_generator(batch_size=10,source=Source.DIGIT_WAVES,target=Target.digits):
    """
    If you set dynamic_pad=True when calling tf.train.batch
    the returned batch will be automatically padded with 0s.
    A lower-level option is to use tf.PaddingFIFOQueue.
    Only apply to a subset of all images at one time.
    """
    maybe_download(source, DATA_DIR)
    if target == Target.speaker: speakers=get_speakers()
    batch_waves = []
    labels = []
    files = os.listdir(path)
    while True:
        random.shuffle(files)
        print("loaded batch of %d files" % len(files))
        for wav in files:
            if not wav.endswith(".wav"):continue
            if target==Target.digits: labels.append(dense_to_one_hot(int(wav[0])))
            elif target==Target.speaker: labels.append(one_hot_from_item(speaker(wav), speakers))
            elif target==Target.first_letter:  label=dense_to_one_hot((ord(wav[0]) - 48) % 32,32)
            else: raise Exception("todo : Target.word label!")
            chunk = load_wav_file(path+wav)
            batch_waves.append(chunk)
            if len(batch_waves) >= batch_size:
                yield batch_waves, labels
                batch_waves = []
                labels = []

if __name__ == "__main__":
    print("downloading speech datasets")
    maybe_download(Source.DIGIT_SPECTROS)
    maybe_download(Source.DIGIT_WAVES)
    maybe_download(Source.NUMBER_IMAGES)
    maybe_download(Source.NUMBER_WAVES)
