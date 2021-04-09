# -*- coding: utf-8 -*-

#%% declaration class and init
import matplotlib.pyplot as plt


class result_graph:

    def __init__(self):
        self.train_loss, self.train_metric = list(), list()
        self.eval_loss, self.eval_metric = list(), list()


#%% make each lists
    def make_list(self, **kwarg):
        if "train_result" in kwarg.keys():
            self.train_loss += kwarg["train_result"].history["loss"]
            self.train_metric += kwarg["train_result"].history['accuracy']
        if "eval_loss" in kwarg.keys():
            self.eval_loss.append(kwarg['eval_loss'])
        if "eval_acc" in kwarg.keys():
            self.eval_metric.append(kwarg['eval_acc'])


#%% draw graph
    def draw_plt_graph(self):

        fig, ax = plt.subplots()
        ax.plot(self.train_loss, 'y', label='train loss')
        ax.plot(self.eval_loss, 'b', label='test loss')

        ax_twinx = ax.twinx()

        ax_twinx.plot(self.train_metric, 'r', label='train accuracy')
        ax_twinx.plot(self.eval_metric, 'g', label='test accuracy')

        ax.set_xlabel('x epochs')
        ax.set_ylabel('y loss')

        ax_twinx.set_ylabel('y accuracy')

        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax_twinx.legend(bbox_to_anchor=(1.05, 0.1), loc='upper left')

        ax.set_title('Result Graph')

        # plt.ylim(0.0, 1.0)

        plt.tight_layout()
        plt.show()

        return


#%% basic graph -> train, validataion
    def draw_train_val_graph(self, history):

        t_acc = history['accuracy']
        v_acc = history['val_accuracy']

        t_loss = history['loss']
        v_loss = history['val_loss']

        fig, ax = plt.subplots()
        ax.plot(t_loss, 'y', label='train loss')
        ax.plot(v_loss, 'b', label='val loss')

        ax_twinx = ax.twinx()

        ax_twinx.plot(t_acc, 'r', label='train accuracy')
        ax_twinx.plot(v_acc, 'g', label='val accuracy')

        ax.set_xlabel('x epochs')
        ax.set_ylabel('y loss')

        ax_twinx.set_ylabel('y accuracy')

        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax_twinx.legend(bbox_to_anchor=(1.05, 0.1), loc='upper left')

        ax.set_title('Result Graph')

        plt.tight_layout()
        plt.show()

        return



#%% __main__
if __name__ == '__main__':
    print('hello, world~!')
