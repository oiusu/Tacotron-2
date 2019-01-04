#!/usr/bin/env python
# encoding: utf-8
'''
@author: chenc
@time: 2019/1/3 4:02 PM
@desc:
'''
import os
import time

from hparams import hparams, hparams_debug_string
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm
from synthesize import prepare_run, get_sentences
from tacotron.synthesize import tacotron_synthesize
from pypinyin import pinyin, lazy_pinyin, Style
import tensorflow as tf
from infolog import log


class initArgs:

    def __init__(self):
            self.GTA= 'True'
            self.checkpoint = 'pretrained/'
            self.hparams = ''
            self.input_dir = 'training_data/'
            self.mels_dir = 'tacotron_output/eval/'
            self.mode = 'eval'
            self.model = 'Tacotron'
            self.name = None
            self.output_dir = 'output/'
            self.speaker_id = None
            self.tacotron_name = None
            self.text_list = 'test.txt'
            self.wavenet_name = None




args = initArgs()
# def loadModel(args):
taco_checkpoint, wave_checkpoint, hparams = prepare_run(args)
output_dir = 'tacotron_' + args.output_dir



try:
    checkpoint_path = tf.train.get_checkpoint_state(taco_checkpoint).model_checkpoint_path
    log('loaded model at {}'.format(checkpoint_path))
except:
    raise RuntimeError('Failed to load checkpoint at {}'.format(taco_checkpoint))

eval_dir = os.path.join(output_dir, 'eval')
log_dir = os.path.join(output_dir, 'logs-eval')

if args.model == 'Tacotron-2':
    assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

# Create output path if it doesn't exist
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

log(hparams_debug_string())
synth = Synthesizer()
synth.load(checkpoint_path, hparams)
print("init finish...")



def get_sentencesByTex(tex):
    if tex != '':
        sentences_py = []
        sentences_result = []
        for sentence in tex:
            sentences_py.append(chatater2pinyin(sentence[-1].split(' ')))
        # sentences_py.append("a1")
        sentences_result.append((" ".join(sentences_py)))

    # else:
    #	sentences = hparams.sentences
    return sentences_result


def chatater2pinyin(labels,is_lazy=0):
	words = []
	for label in labels:
		#print(label,pinyin(label)[0])
		if is_lazy:
			phrase = ''.join(pinyin(lazy_label))
		else:
			phrase = [x[0] for x in pinyin(label,style=Style.TONE3)]
			phrase = ''.join(phrase)
		words.append(phrase)
	return ' '.join(words)


def tacotron_synthesize(logId , sentences):
    # Set inputs batch wise
    sentences = [sentences[i: i + hparams.tacotron_synthesis_batch_size] for i in
             range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]
    log("logId={} , sentences={}".format(logId,sentences))
    # basenames = logId
    # texts = sentences
    # synth.synthesizev1(texts, basenames, eval_dir, log_dir, None)
    # mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)
    t1 = time.time()
    for i, texts in enumerate(tqdm(sentences)):

        # basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
        basenames = [logId]
        wavPaths = synth.synthesizev1(texts, basenames, eval_dir, log_dir, None)
    t2 = time.time()
    log('logId={} , synthesized mel spectrograms at {} cost time={}'.format(logId,eval_dir,(t2-t1)))


    # with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
    #     for i, texts in enumerate(tqdm(sentences)):
    #         start = time.time()
    #         basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
    #         # basenames = logId
    #         mel_filenames, speaker_ids = synth.synthesizev1(texts, basenames, eval_dir, log_dir, None)
    #
    #         for elems in zip(texts, mel_filenames, speaker_ids):
    #             file.write('|'.join([str(x) for x in elems]) + '\n')
    # log('synthesized mel spectrograms at {}'.format(eval_dir))
    return wavPaths[0]


# def run_eval_bak(args, checkpoint_path, output_dir, hparams, sentences):
# 	eval_dir = os.path.join(output_dir, 'eval')
# 	log_dir = os.path.join(output_dir, 'logs-eval')
#
# 	if args.model == 'Tacotron-2':
# 		assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)
#
# 	#Create output path if it doesn't exist
# 	os.makedirs(eval_dir, exist_ok=True)
# 	os.makedirs(log_dir, exist_ok=True)
# 	os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
# 	os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
#
# 	log(hparams_debug_string())
# 	synth = Synthesizer()
# 	synth.load(checkpoint_path, hparams)
#
# 	#Set inputs batch wise
# 	sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]
#
# 	log('Starting Synthesis')
# 	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
# 		for i, texts in enumerate(tqdm(sentences)):
# 			start = time.time()
# 			basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
# 			mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)
#
# 			for elems in zip(texts, mel_filenames, speaker_ids):
# 				file.write('|'.join([str(x) for x in elems]) + '\n')
# 	log('synthesized mel spectrograms at {}'.format(eval_dir))
# 	return eval_dir



def synthesizeBytex(logId,tex):


    sentences = get_sentencesByTex(tex)
    wavPath = tacotron_synthesize(logId, sentences)
    return wavPath





