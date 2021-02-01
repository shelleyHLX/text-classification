import argparse, sys, os, time, logging, threading, traceback
import numpy as np
import tensorflow as tf
import _pickle as pkl
import sys
from multiprocessing import Queue, Process

from caps_attn_hierarchical.Config import Config
from caps_attn_hierarchical.model import model
from caps_attn_hierarchical.data_iterator import TextIterator, preparedata
from caps_attn_hierarchical.dataprocess.vocab import Vocab
from caps_attn_hierarchical import utils

_REVISION = 'None'

parser = argparse.ArgumentParser(description="training options")

parser.add_argument('--load-config', action='store_true', dest='load_config', default='--weight-path')
parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
parser.add_argument('--weight-path', action='store', dest='weight_path', default='../savings/imdb/', required=False)
parser.add_argument('--restore-ckpt', action='store_true', dest='restore_ckpt', default=False)
parser.add_argument('--retain-gpu', action='store_true', dest='retain_gpu', default=False)
parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)

args = parser.parse_args()

DEBUG = args.debug_enable
if not DEBUG:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def debug(s):
    if DEBUG:
        print(s)
    pass


class Train(object):
    def __init__(self, args):
        if utils.valid_entry(args.weight_path) and not args.restore_ckpt\
                and args.train_test != 'test':
            raise ValueError('process running or finished')

        gpu_lock = threading.Lock()
        gpu_lock.acquire()
        def retain_gpu():
            if args.retain_gpu:
                with tf.Session():
                    gpu_lock.acquire()
            else:
                pass

        lockThread = threading.Thread(target=retain_gpu)
        lockThread.start()
        try:
            self.args = args
            config = Config()

            self.args = args
            self.weight_path = args.weight_path

            if args.load_config == False:
                config.saveConfig(self.weight_path + '/config')
                print('default configuration generated, please specify --load-config and run again.')
                gpu_lock.release()
                lockThread.join()
                sys.exit()
            else:
                if os.path.exists(self.weight_path + '/config'):
                    config.loadConfig(self.weight_path + '/config')
                else:
                    raise ValueError('No config file in %s' % self.weight_path)

            if config.revision != _REVISION:
                raise ValueError('revision dont match: %s over %s' % (config.revision, _REVISION))

            vocab = Vocab()
            vocab.load_vocab_from_file(os.path.join(config.datapath, 'vocab.pkl'))
            config.vocab_dict = vocab.word_to_index
            with open(os.path.join(config.datapath, 'label2id.pkl'), 'rb') as fd:
                _ = pkl.load(fd)
                config.id2label = pkl.load(fd)
                # print(config.id2label)
                _ = pkl.load(fd)
                config.id2weight = pkl.load(fd)
                # print(config.id2weight)
                """
                {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10'}
{0: 2.116198234610293, 1: 2.481059999198275, 2: 1.6450506516423342, 3: 1.1190706625654634,
                """
                # exit(0)
            config.class_num = len(config.id2label)
            self.config = config

            self.train_data = TextIterator(os.path.join(config.datapath, 'trainset.pkl'), self.config.batch_sz,
                                           bucket_sz=self.config.bucket_sz, shuffle=True)
            # print(self.train_data.keys())
            # exit(0)
            config.n_samples = self.train_data.num_example

            self.dev_data = TextIterator(os.path.join(config.datapath, 'devset.pkl'), self.config.batch_sz,
                                         bucket_sz=self.config.bucket_sz, shuffle=False)

            self.test_data = TextIterator(os.path.join(config.datapath, 'testset.pkl'), self.config.batch_sz,
                                         bucket_sz=self.config.bucket_sz, shuffle=False)

            self.data_q = Queue(10)

            self.model = model(config)
            # print('config ', config)
            # exit(0)

        except Exception as e:
            traceback.print_exc()
            gpu_lock.release()
            lockThread.join()
            exit()

        gpu_lock.release()
        lockThread.join()
        if utils.valid_entry(args.weight_path) and not args.restore_ckpt\
                and args.train_test != 'test':
            raise ValueError('process running or finished')

    def get_epoch(self, sess):
        epoch = sess.run(self.model.on_epoch)
        return epoch

    def run_epoch(self, sess, input_data: TextIterator, verbose=100):
        """Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_steps = input_data.num_example // input_data.batch_sz
        total_loss = []
        total_w_loss = []
        total_ce_loss = []
        collect_time = []
        collect_data_time = []
        accuracy_collect = []
        step = -1
        dataset = [o for o in input_data]
        producer = Process(target=preparedata,
                           args=(dataset, self.data_q, self.config.max_snt_num,
                                 self.config.max_wd_num, self.config.id2weight))
        producer.start()
        while True:
            step += 1
            start_stamp = time.time()
            # self.ph_input([[[], [], ...], ...], [文档(句子,句子,...),文档]), self.ph_labels,
            # self.ph_sNum([句子个数,...], 有用)一个文档的句子个数, self.ph_wNum([[一个句子词个数,...], ...]), self.ph_sample_weights,
            # 一个batch的ph_input的第二维,不同batch可能不同.(64, 7, 30) 30不变
            data_batch = self.data_q.get()
            if data_batch is None:
                break
            # print(type(data_batch))
            # print(len(data_batch))
            # print(type(data_batch[0]))
            print(data_batch[0].shape)  # (64, 25, 30)
            for i in range(64):
                print(data_batch[0][i].shape)
            print(type(data_batch[1]))
            print(data_batch[1].shape)
            print(type(data_batch[2]))
            print(data_batch[2])
            # print(type(data_batch[3]))
            # print(data_batch[3].shape)
            print(data_batch[0][60])
            print(data_batch[2][60])
            print(data_batch[3][60])
            continue
        #
        #     feed_dict = self.model.create_feed_dict(data_batch=data_batch, train=True)
        #     # print('() '*100)
        #     # for i in feed_dict.keys():
        #     #     print(i)
        #     #     print(type(feed_dict[i]))
        #     #     print(feed_dict[i][0])
        #         # break
        #     # break
        #     # print(feed_dict['Tensor("ph_input:0", shape=(?, ?, ?), dtype=int32)'].shape)
        #     # exit(0)
        #
        #     data_stamp = time.time()
        #     (accuracy, global_step, summary, opt_loss, w_loss, ce_loss, lr, _
        #      ) = sess.run([self.model.accuracy, self.model.global_step, self.merged,
        #                    self.model.opt_loss, self.model.w_loss, self.model.ce_loss,
        #                    self.model.learning_rate, self.model.train_op],
        #                   feed_dict=feed_dict)
        #     self.train_writer.add_summary(summary, global_step)
        #     self.train_writer.flush()
        #
        #     end_stamp = time.time()
        #
        #     collect_time.append(end_stamp-start_stamp)
        #     collect_time.append(data_stamp - start_stamp)
        #     accuracy_collect.append(accuracy)
        #     total_loss.append(opt_loss)
        #     total_w_loss.append(w_loss)
        #     total_ce_loss.append(ce_loss)
        #
        #     if verbose and step % verbose == 0:
        #         sys.stdout.write('\r%d / %d : opt_loss = %.4f, w_loss = %.4f, ce_loss = %.4f, %.3fs/iter, %.3fs/batch, '
        #                          'lr = %f, accu = %.4f, b_sz = %d' % (
        #             step, total_steps, float(np.mean(total_loss[-verbose:])), float(np.mean(total_w_loss[-verbose:])),
        #             float(np.mean(total_ce_loss[-verbose:])), float(np.mean(collect_time)), float(np.mean(collect_data_time)), lr,
        #             float(np.mean(accuracy_collect[-verbose:])), input_data.batch_sz))
        #         collect_time = []
        #         sys.stdout.flush()
        #         utils.write_status(self.weight_path)
        # producer.join()
        # sess.run(self.model.on_epoch_accu)
        exit(0)
        return np.mean(total_ce_loss), np.mean(total_loss), np.mean(accuracy_collect)

    def fit(self, sess, input_data :TextIterator, verbose=10):
        """
        Fit the model.

        Args:
            sess: tf.Session() object
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """

        total_steps = input_data.num_example // input_data.batch_sz
        total_loss = []
        total_ce_loss = []
        collect_time = []
        step = -1
        dataset = [o for o in input_data]
        producer = Process(target=preparedata,
                           args=(dataset, self.data_q, self.config.max_snt_num,
                                 self.config.max_wd_num, self.config.id2weight))
        producer.start()
        while True:
            step += 1
            data_batch = self.data_q.get()
            if data_batch is None:
                break
            feed_dict = self.model.create_feed_dict(data_batch=data_batch, train=False)

            start_stamp = time.time()
            (global_step, summary, ce_loss, opt_loss,
             ) = sess.run([self.model.global_step, self.merged, self.model.ce_loss,
                           self.model.opt_loss], feed_dict=feed_dict)

            self.test_writer.add_summary(summary, step+global_step)
            self.test_writer.flush()

            end_stamp = time.time()
            collect_time.append(end_stamp - start_stamp)
            total_ce_loss.append(ce_loss)
            total_loss.append(opt_loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r%d / %d: ce_loss = %f, opt_loss = %f,  %.3fs/iter' % (
                    step, total_steps, float(np.mean(total_ce_loss[-verbose:])),
                    float(np.mean(total_loss[-verbose:])), float(np.mean(collect_time))))
                collect_time = []
                sys.stdout.flush()
        print('\n')
        producer.join()
        return np.mean(total_ce_loss), np.mean(total_loss)

    def predict(self, sess, input_data: TextIterator, verbose=10):
        """
        Args:
            sess: tf.Session() object
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_steps = input_data.num_example // input_data.batch_sz
        collect_time = []
        collect_pred = []
        label_id = []
        step = -1
        dataset = [o for o in input_data]
        producer = Process(target=preparedata,
                           args=(dataset, self.data_q, self.config.max_snt_num,
                                 self.config.max_wd_num, self.config.id2weight))
        producer.start()
        while True:
            step += 1
            data_batch = self.data_q.get()
            if data_batch is None:
                break
            feed_dict = self.model.create_feed_dict(data_batch=data_batch, train=False)

            start_stamp = time.time()
            pred = sess.run(self.model.prediction, feed_dict=feed_dict)
            end_stamp = time.time()
            collect_time.append(end_stamp - start_stamp)

            collect_pred.append(pred)
            label_id += data_batch[1].tolist()
            if verbose and step % verbose == 0:
                sys.stdout.write('\r%d / %d: , %.3fs/iter' % (step, total_steps, float(np.mean(collect_time))))
                collect_time = []
                sys.stdout.flush()
        print('\n')
        producer.join()
        res_pred = np.concatenate(collect_pred, axis=0)
        return res_pred, label_id

    def test_case(self, sess, data, onset='VALIDATION'):
        print('#' * 20, 'ON ' + onset + ' SET START ef', '#' * 20)
        print("=" * 10 + ' '.join(sys.argv) + "=" * 10)
        epoch = self.get_epoch(sess)
        ce_loss, opt_loss = self.fit(sess, data)
        pred, label = self.predict(sess, data)

        (prec, recall, overall_prec, overall_recall, _
         ) = utils.calculate_confusion_single(pred, label, len(self.config.id2label))

        utils.print_confusion_single(prec, recall, overall_prec, overall_recall, self.config.id2label)
        accuracy = utils.calculate_accuracy_single(pred, label)

        print('%d th Epoch -- Overall %s accuracy is: %f' % (epoch, onset, accuracy))
        logging.info('%d th Epoch -- Overall %s accuracy is: %f' % (epoch, onset, accuracy))

        print('%d th Epoch -- Overall %s ce_loss is: %f, opt_loss is: %f' % (epoch, onset, ce_loss, opt_loss))
        logging.info('%d th Epoch -- Overall %s ce_loss is: %f, opt_loss is: %f' % (epoch, onset, ce_loss, opt_loss))
        print('#' * 20, 'ON ' + onset + ' SET END ', '#' * 20)
        return accuracy, ce_loss

    def train_run(self):
        logging.info('Training start')
        logging.info("Parameter count is: %d" % self.model.param_cnt)
        if not args.restore_ckpt:
            self.remove_file(self.args.weight_path + '/summary.log')
        saver = tf.train.Saver(max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_train/', sess.graph)
            self.test_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_test/')
            sess.run(tf.global_variables_initializer())
            if args.restore_ckpt:
                saver.restore(sess, tf.train.latest_checkpoint(self.args.weight_path))
            best_loss = np.Inf
            best_accuracy = 0
            best_val_epoch = self.get_epoch(sess)

            for _ in range(self.config.max_epochs):
                epoch = self.get_epoch(sess)
                print("=" * 20 + "Epoch ", epoch, "=" * 20)
                ce_loss, opt_loss, accuracy = self.run_epoch(sess, self.train_data, verbose=10)
                print('')
                print("Mean ce_loss in %dth epoch is: %f, Mean ce_loss is: %f,"%(epoch, ce_loss, opt_loss))
                print('Mean training accuracy is : %.4f' % accuracy)
                logging.info('Mean training accuracy is : %.4f' % accuracy)
                logging.info("Mean ce_loss in %dth epoch is: %f, Mean ce_loss is: %f,"%(epoch, ce_loss, opt_loss))
                print('=' * 50)
                val_accuracy, val_loss = self.test_case(sess, self.dev_data, onset='VALIDATION')
                test_accuracy, test_loss = self.test_case(sess, self.test_data, onset='TEST')
                self.save_loss_accu(self.args.weight_path + '/summary.log', train_loss=ce_loss,
                                    valid_loss=val_loss, test_loss=test_loss,
                                    valid_accu=val_accuracy, test_accu=test_accuracy, epoch=epoch)
                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(self.args.weight_path):
                        os.makedirs(self.args.weight_path)
                    logging.info('best epoch is %dth epoch' % best_val_epoch)
                    saver.save(sess, self.args.weight_path + 'model.ckpt')
                else:
                    b_sz = self.train_data.batch_sz//2
                    max_b_sz = max([b_sz, self.config.batch_sz_min])
                    buck_sz = self.train_data.bucket_sz * 2
                    buck_sz = min([self.train_data.num_example, buck_sz])
                    self.train_data.batch_sz = max_b_sz
                    self.train_data.bucket_sz = buck_sz

                if epoch - best_val_epoch > self.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
        utils.write_status(self.weight_path, finished=True)
        logging.info("Training complete")

    def test_run(self):
        saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            self.merged = tf.summary.merge_all()
            self.test_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_test')

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(self.args.weight_path))
# tf.train.latest_checkpoint(ckpt_path)
            self.test_case(sess, self.test_data, onset='TEST')

    def main_run(self):

        if not os.path.exists(self.args.weight_path):
            os.makedirs(self.args.weight_path)
        logFile = self.args.weight_path + '/run.log'

        if self.args.train_test == "train":
            try:
                os.remove(logFile)
            except OSError:
                pass
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
            # debug('_main_run_')
            self.train_run()
            self.test_run()
        else:
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
            self.test_run()

    @staticmethod
    def save_loss_accu(fileName, train_loss, valid_loss,
                       test_loss, valid_accu, test_accu, epoch):
        with open(fileName, 'a') as fd:
            fd.write('%3d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' %
                     (epoch, train_loss, valid_loss,
                      test_loss, valid_accu, test_accu))

    @staticmethod
    def remove_file(fileName):
        if os.path.exists(fileName):
            os.remove(fileName)

if __name__ == '__main__':
    trainer = Train(args)
    trainer.main_run()

"""
[[   17   244  3037   506   301   481  6838    16   312   534     6    65
     12  4445    59     5  1084  3303   105     7     0     0     0     0
      0     0     0     0     0     0]
 [   17    77 18605     5  4295    22     9   312    56   116    60   172
    882     9   127   100  5937   189     5   406    13     5  2232     7
      0     0     0     0     0     0]
 [   17   114  2497    11    18   745    61    81     6    10   406    42
     41    89   362    71   112 24831   266    12     6     8   112     5
   5826   647    10   220   311   611]
 [   13     5   376    10   481  6838    16   119     6    18   589    66
   1283   169     5   309  1792    10     5    23     7     0     0     0
      0     0     0     0     0     0]
 [   14    16   309   804     6     9  2358   679    20    54   697    10
     26   200     6   589    66   261    11  1243    18   658   169    26
    200 16416     7     0     0     0]
 [   21    14    16    35    77    55    15    19    71     5  2929    10
      9   128    15   188     8  6205   103    14    16     5   670    10
    876     6  7445     6   886     6]
 [    5   221   534    15  9019    18    12   422    66    11     9   240
    103    14   420    10   589     9   127  7540     8  1094     8  3101
     25     5   381     8 11454    16]
 [    8     5   227    13   770    33   173    58    22    68   597    67
      9   127 18855   100     7     0     0     0     0     0     0     0
      0     0     0     0     0     0]
 [   21    49    27  1717     9   494     6   509    22     5    96   393
      6    15    17    93    34  1459    84     7     0     0     0     0
      0     0     0     0     0     0]
 [   14    16   220    51    37    10   158 53156   376   678   317    11
  16069     6   229 12907    20    15  1345    10     5  2088  2943   182
    652    56  6035     1   110     7]
 [  150   406    33    71    32     5 15654     6     8    14    16    49
    121    47    75    99 17022     6  1602     6   186    25    13     9
   4278   108    24     6     8  4798]
 [   14    16     9    58     6   506    71  1080    23    55  4905  5090
    189   219     6   440     6     8 13977    13   158  6901  5569   162
      6     8     9   248    10  1560]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0     0     0]]
[20 24 44 21 27 45 44 17 20 30 40 49  0]
"""