#!/usr/bin/env python
# Official implementation code for "Lung Nodule Detection and Classification from Thorax CT-Scan Using RetinaNet with Transfer Learning" and "Lung Nodule Texture Detection and Classification Using 3D CNN."
# Adapted from of [medicaldetectiontoolkit](https://github.com/pfjaeger/medicaldetectiontoolkit) and [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import logging
import subprocess
import os
import torch
from collections import OrderedDict
import plotting
import sys
import importlib.util
import pandas as pd


def get_logger(exp_dir):
    """
    creates logger instance. writing out info to file and to terminal.
    :param exp_dir: experiment directory, where exec.log file is stored.
    :return: logger instance.
    """

    logger = logging.getLogger('medicaldetectiontoolkit')
    logger.setLevel(logging.DEBUG)
    log_file = exp_dir + '/exec.log'
    hdlr = logging.FileHandler(log_file)
    print('Logging to {}'.format(log_file))
    logger.addHandler(hdlr)
    logger.addHandler(ColorHandler())
    logger.propagate = False
    return logger



def prep_exp(dataset_path, exp_path, server_env, use_stored_settings=True, is_training=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param server_env: boolean flag. pass to configs script for cloud deployment.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :param is_training: boolean flag. distinguishes train vs. inference mode.
    :return:
    """

    if is_training:

        # the first process of an experiment creates the directories and copies the config to exp_path.
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
            os.mkdir(os.path.join(exp_path, 'plots'))
            subprocess.call('cp {} {}'.format(os.path.join(dataset_path, 'configs.py'), os.path.join(exp_path, 'configs.py')), shell=True)
            subprocess.call('cp {} {}'.format('default_configs.py', os.path.join(exp_path, 'default_configs.py')), shell=True)


        if use_stored_settings:
            subprocess.call('cp {} {}'.format('default_configs.py', os.path.join(exp_path, 'default_configs.py')), shell=True)
            cf_file = import_module('cf', os.path.join(exp_path, 'configs.py'))
            cf = cf_file.configs(server_env)
            # only the first process copies the model selcted in configs to exp_path.
            if not os.path.isfile(os.path.join(exp_path, 'model.py')):
                subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(exp_path, 'model.py')), shell=True)
                subprocess.call('cp {} {}'.format(os.path.join(cf.backbone_path), os.path.join(exp_path, 'backbone.py')), shell=True)

            # copy the snapshot model scripts from exp_dir back to the source_dir as tmp_model / tmp_backbone.
            tmp_model_path = os.path.join(cf.source_dir, 'models', 'tmp_model.py')
            tmp_backbone_path = os.path.join(cf.source_dir, 'models', 'tmp_backbone.py')
            subprocess.call('cp {} {}'.format(os.path.join(exp_path, 'model.py'), tmp_model_path), shell=True)
            subprocess.call('cp {} {}'.format(os.path.join(exp_path, 'backbone.py'), tmp_backbone_path), shell=True)
            cf.model_path = tmp_model_path
            cf.backbone_path = tmp_backbone_path

        else:
            # run training with source code info and copy snapshot of model to exp_dir for later testing (overwrite scripts if exp_dir already exists.)
            cf_file = import_module('cf', os.path.join(dataset_path, 'configs.py'))
            cf = cf_file.configs(server_env)
            subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(exp_path, 'model.py')), shell=True)
            subprocess.call('cp {} {}'.format(cf.backbone_path, os.path.join(exp_path, 'backbone.py')), shell=True)
            subprocess.call('cp {} {}'.format('default_configs.py', os.path.join(exp_path, 'default_configs.py')), shell=True)
            subprocess.call('cp {} {}'.format(os.path.join(dataset_path, 'configs.py'), os.path.join(exp_path, 'configs.py')), shell=True)

    else:
        # for testing copy the snapshot model scripts from exp_dir back to the source_dir as tmp_model / tmp_backbone.
        cf_file = import_module('cf', os.path.join(exp_path, 'configs.py'))
        cf = cf_file.configs(server_env)
        if cf.hold_out_test_set:
            cf.pp_data_path = cf.pp_test_data_path
            cf.pp_name = cf.pp_test_name
        tmp_model_path = os.path.join(cf.source_dir, 'models', 'tmp_model.py')
        tmp_backbone_path = os.path.join(cf.source_dir, 'models', 'tmp_backbone.py')
        subprocess.call('cp {} {}'.format(os.path.join(exp_path, 'model.py'), tmp_model_path), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_path, 'backbone.py'), tmp_backbone_path), shell=True)
        cf.model_path = tmp_model_path
        cf.backbone_path = tmp_backbone_path

    cf.exp_dir = exp_path
    cf.test_dir = os.path.join(cf.exp_dir, 'test')
    cf.plot_dir = os.path.join(cf.exp_dir, 'plots')
    cf.experiment_name = exp_path.split("/")[-1]
    cf.server_env = server_env
    cf.created_fold_id_pickle = False

    return cf



def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



class ModelSelector:
    '''
    saves a checkpoint after each epoch as 'last_state' (can be loaded to continue interrupted training).
    saves the top-k (k=cf.save_n_models) ranked epochs. In inference, predictions of multiple epochs can be ensembled to improve performance.
    '''

    def __init__(self, cf, logger):

        self.cf = cf
        self.saved_epochs = [-1] * cf.save_n_models
        self.logger = logger

    def run_model_selection(self, net, optimizer, monitor_metrics, epoch):

        # take the mean over all selection criteria in each epoch
        non_nan_scores = np.mean(np.array([[0 if ii is None else ii for ii in monitor_metrics['val'][sc]] for sc in self.cf.model_selection_criteria]), 0)
        print('non none scores:', non_nan_scores)
        epochs_scores = [ii for ii in non_nan_scores[1:]]
        # ranking of epochs according to model_selection_criterion
        epoch_ranking = np.argsort(epochs_scores)[::-1] + 1 #epochs start at 1
        # if set in configs, epochs < min_save_thresh are discarded from saving process.
        epoch_ranking = epoch_ranking[epoch_ranking >= self.cf.min_save_thresh]

        # check if current epoch is among the top-k epchs.
        if epoch in epoch_ranking[:self.cf.save_n_models]:
            torch.save(net.state_dict(), os.path.join(self.cf.fold_dir, '{}_best_params.pth'.format(epoch)))
            # save epoch_ranking to keep info for inference.
            np.save(os.path.join(self.cf.fold_dir, 'epoch_ranking'), epoch_ranking[:self.cf.save_n_models])
            self.logger.info(
                "saving current epoch {} at rank {}".format(epoch, np.argwhere(epoch_ranking == epoch)))
            # delete params of the epoch that just fell out of the top-k epochs.
            for se in [int(ii.split('_')[0]) for ii in os.listdir(self.cf.fold_dir) if 'best_params' in ii]:
                if se in epoch_ranking[self.cf.save_n_models:]:
                    subprocess.call('rm {}'.format(os.path.join(self.cf.fold_dir, '{}_best_params.pth'.format(se))), shell=True)
                    self.logger.info('deleting epoch {} at rank {}'.format(se, np.argwhere(epoch_ranking == se)))

        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(state, os.path.join(self.cf.fold_dir, 'last_state.pth'))



def load_checkpoint(checkpoint_path, net, optimizer):

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def load_rgb_weight(rgb_weights_path, net):

    # weight_i3d = torch.load(rgb_weights_path)
    net.Fpn.load_state_dict(torch.load(rgb_weights_path), strict=False)
    # net.load_state_dict(checkpoint, strict=False)
    status = 'Succeeded'
    return status, net

def prepare_monitoring(cf):
    """
    creates dictionaries, where train/val metrics are stored.
    """
    metrics = {}
    # first entry for loss dict accounts for epoch starting at 1.
    metrics['train'] = OrderedDict()
    metrics['val'] = OrderedDict()
    metric_classes = []
    if 'rois' in cf.report_score_level:
        metric_classes.extend([v for k, v in cf.class_dict.items()])
    if 'patient' in cf.report_score_level:
        metric_classes.extend(['patient'])
    for cl in metric_classes:
        metrics['train'][cl + '_ap'] = [None]
        metrics['val'][cl + '_ap'] = [None]
        if cl == 'patient':
            metrics['train'][cl + '_auc'] = [None]
            metrics['val'][cl + '_auc'] = [None]

    metrics['train']['monitor_values'] = [[] for _ in range(cf.num_epochs + 1)]
    metrics['val']['monitor_values'] = [[] for _ in range(cf.num_epochs + 1)]

    # generate isntance of monitor plot class.
    TrainingPlot = plotting.TrainingPlot_2Panel(cf)

    return metrics, TrainingPlot



def create_csv_output(cf, logger, results_list):
    """
    Write out test set predictions to .csv file. output format is one line per patient:
    PatientID score pred_class x y w h score pred_class x y w h .....
    :param results_list: [[patient_results, patient_id], [patient_results, patient_id], ...]
    """
    logger.info('creating csv output file at {}'.format(os.path.join(cf.exp_dir, 'output.csv')))
    submission_df = pd.DataFrame(columns=['patientID', 'PredictionString'])
    for r in results_list:
        pid = r[1]
        prediction_string = ''
        for box in r[0][0]:
            coords = box['box_coords']
            score = box['box_score']
            pred_class = box['box_pred_class_id']

            if score >= cf.min_det_thresh:
                x = coords[1] #* cf.pp_downsample_factor
                y = coords[0] #* cf.pp_downsample_factor
                width = (coords[3] - coords[1]) #* cf.pp_downsample_factor
                height = (coords[2] - coords[0]) #* cf.pp_downsample_factor
                if len(coords) == 6:
                    z = coords[4]
                    depth = (coords[5] - coords[4])
                    prediction_string += '{} {} {} {} {} {} {} {}'.format(score, pred_class, x, y, z, width, height, depth)
                else:
                    prediction_string += '{} {} {} {} {} {} '.format(score, pred_class, x, y, width, height)

        if prediction_string == '':
            prediction_string = None
        submission_df.loc[len(submission_df)] = [pid, prediction_string]
    submission_df.to_csv(os.path.join(cf.exp_dir, 'output.csv'), index=False)



class _AnsiColorizer(object):
    """
    A colorizer is an object that loosely wraps around a stream, allowing
    callers to write text to the stream in a particular color.

    Colorizer classes must implement C{supported()} and C{write(text, color)}.
    """
    _colors = dict(black=30, red=31, green=32, yellow=33,
                   blue=34, magenta=35, cyan=36, white=37, default=39)

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def supported(cls, stream=sys.stdout):
        """
        A class method that returns True if the current platform supports
        coloring terminal output using this method. Returns False otherwise.
        """
        if not stream.isatty():
            return False  # auto color only on TTYs
        try:
            import curses
        except ImportError:
            return False
        else:
            try:
                try:
                    return curses.tigetnum("colors") > 2
                except curses.error:
                    curses.setupterm()
                    return curses.tigetnum("colors") > 2
            except:
                raise
                # guess false in case of error
                return False

    def write(self, text, color):
        """
        Write the given text to the stream in the given color.

        @param text: Text to be written to the stream.

        @param color: A string label for a color. e.g. 'red', 'white'.
        """
        color = self._colors[color]
        self.stream.write('\x1b[%sm%s\x1b[0m' % (color, text))



class ColorHandler(logging.StreamHandler):


    def __init__(self, stream=sys.stdout):
        super(ColorHandler, self).__init__(_AnsiColorizer(stream))

    def emit(self, record):
        msg_colors = {
            logging.DEBUG: "green",
            logging.INFO: "default",
            logging.WARNING: "red",
            logging.ERROR: "red"
        }
        color = msg_colors.get(record.levelno, "blue")
        self.stream.write(record.msg + "\n", color)

