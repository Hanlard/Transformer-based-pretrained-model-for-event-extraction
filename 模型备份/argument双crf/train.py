"""
2020/1/29修改
加入了提前终止
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from model import Net

from data_load import ACE2005Dataset, pad, all_triggers, all_entities, all_postags, all_arguments, tokenizer, idx2trigger
from utils import calc_metric, find_triggers

from utils import report_to_telegram
from eval import eval
import warnings
warnings.filterwarnings('ignore')


def train(model, iterator, optimizer, hp):
    model.train()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
    # 角色
    # argument_keys：（正确）预测触发词 - 正确实体
    # arguments_2d：正确触发词 - 正确角色
    # 正确实体
    # arguments_y_2d：输入CRF的标签数据[dim0, seq_len]
    # argument_hat_1d: CRF计算结果
    # argument_hat_2d：根据argument_keys和argument_hat_1d写成字典格式
    #
    # 触发词
    # trigger_hat_2d：CRF预测触发词
    # triggers_y_2d：正确触发词
    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm = batch
        optimizer.zero_grad()
        ## crf_loss， 触发词标签， 预测触发词， 实体-事件拼接张量， （7维元组）
        trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                      postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                                                                      triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, adjm=adjm)

        if len(argument_keys) > 0:
            argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(argument_hidden, argument_keys, arguments_2d, adjm)
            # argument_loss = criterion(argument_logits, arguments_y_1d)
            loss =  trigger_loss +  hp.LOSS_alpha* argument_loss
            # if i == 0:

            #     print("=====sanity check for triggers======")
            #     print('triggers_y_2d[0]:', triggers_y_2d[0])
            #     print("trigger_hat_2d[0]:", trigger_hat_2d[0])

            #     print("=======================")

            #     print("=====sanity check for arguments======")
            #     print('arguments_y_2d[0]:', arguments_y_2d[0])
            #     print('argument_hat_1d[0]:', argument_hat_1d[0])
            #     print("arguments_2d[0]:", arguments_2d)
            #     print("argument_hat_2d[0]:", argument_hat_2d)
            #     print("=======================")

        else:
            loss = trigger_loss

        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        loss.backward()

        optimizer.step()

        # if i == 0:
        #     print("=====sanity check======")
        #     print("tokens_x_2d[0]:", tokenizer.convert_ids_to_tokens(tokens_x_2d[0])[:seqlens_1d[0]])
        #     print("entities_x_3d[0]:", entities_x_3d[0][:seqlens_1d[0]])
        #     print("postags_x_2d[0]:", postags_x_2d[0][:seqlens_1d[0]])
        #     print("head_indexes_2d[0]:", head_indexes_2d[0][:seqlens_1d[0]])
        #     print("triggers_2d[0]:", triggers_2d[0])
        #     print("triggers_y_2d[0]:", triggers_y_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
        #     print('trigger_hat_2d[0]:', trigger_hat_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
        #     print("seqlens_1d[0]:", seqlens_1d[0])
        #     print("arguments_2d[0]:", arguments_2d[0])
        #     print("=======================")

        #### 训练精度评估 ####
        words_all.extend(words_2d)
        triggers_all.extend(triggers_2d)
        triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
        arguments_all.extend(arguments_2d)

        if len(argument_keys) > 0:
            arguments_hat_all.extend(argument_hat_2d)
        else:
            batch_size = len(arguments_2d)
            argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
            arguments_hat_all.extend(argument_hat_2d)

        for ii, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(
                zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(ii, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(ii, *item) for item in find_triggers(triggers_hat)])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_true.append((ii, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_pred.append((ii, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

        if i % 100 == 0:  # monitoring
            trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
            argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
            ## 100step 清零
            words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
            triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
            #########################
            if  len(argument_keys) > 0:
                print("【识别到事件】step: {}, loss: {:.3f}, trigger_loss:{:.3f}, argument_loss:{:.3f}".format(i, loss.item(), trigger_loss.item(), argument_loss.item()),
                      '【trigger】 P={:.3f}  R={:.3f}  F1={:.3f}'.format(trigger_p, trigger_r, trigger_f1),
                      '【argument】 P={:.3f}  R={:.3f}  F1={:.3f}'.format(argument_p, argument_r, argument_f1)
                      )
            else:
                print("【未识别到事件】step: {}, loss: {:.3f} ".format(i, loss.item()),
                      '【trigger】 P={:.3f}  R={:.3f}  F1={:.3f}'.format(trigger_p, trigger_r, trigger_f1)
                      )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--trainset", type=str, default="data/train_all.json")
    parser.add_argument("--devset", type=str, default="data/dev.json")
    parser.add_argument("--testset", type=str, default="data/test.json")
    parser.add_argument("--LOSS_alpha", type=float, default=1.0)
    parser.add_argument("--telegram_bot_token", type=str, default="")
    parser.add_argument("--telegram_chat_id", type=str, default="")
    if os.name == "nt":
        parser.add_argument("--model_path", type=str, default="C:\Hanlard\\NLP\模型\事件抽取\\bert-event-extraction-master\save_model\latest_model.pt")
        parser.add_argument("--batch_size", type=int, default=4)
    else:
        parser.add_argument("--model_path", type=str, default="/content/drive/My Drive/Colab Notebooks/模型/事件抽取/bert-event-extraction-master/save_model/latest_model.pt")
        parser.add_argument("--batch_size", type=int, default=16)
    hp = parser.parse_args()
    device = 'cuda' #if torch.cuda.is_available() else 'cpu'
    print("==========超参==========")
    print(hp)

    if os.path.exists(hp.model_path):
        print('=======载入模型=======')
        model = torch.load(hp.model_path)
    else:
        print("=======初始化模型======")
        model = Net(
            device=device,
            trigger_size=len(all_triggers),
            entity_size=len(all_entities),
            all_postags=len(all_postags),
            argument_size=len(all_arguments)
        )
        if device == 'cuda':
            model = model.cuda()

        model = nn.DataParallel(model)

    train_dataset = ACE2005Dataset(hp.trainset)
    dev_dataset = ACE2005Dataset(hp.devset)
    test_dataset = ACE2005Dataset(hp.testset)

    samples_weight = train_dataset.get_samples_weight()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 sampler=sampler,
                                 num_workers=4,
                                 collate_fn=pad)
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.l2)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(ignore_index=0)# size_average = True

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    if not os.path.exists( os.path.split(hp.model_path)[0] ):
        os.makedirs(os.path.split(hp.model_path)[0])

    _, _, argument_f1_test = eval(model, test_iter, os.path.join(hp.logdir,'0') + '_test')
    best_f1 = max(0,argument_f1_test)
    no_gain_rc = 0#效果不增加代数
    for epoch in range(1, hp.n_epochs + 1):
        train(model, train_iter, optimizer, hp)

        fname = os.path.join(hp.logdir, str(epoch))

        print(f"=========eval dev at epoch={epoch}=========")
        metric_dev,trigger_f1_dev, argument_f1_dev = eval(model, dev_iter, fname + '_dev')

        print(f"=========eval test at epoch={epoch}=========")
        metric_test,trigger_f1_test, argument_f1_test = eval(model, test_iter, fname + '_test')

        if hp.telegram_bot_token:
            report_to_telegram('[epoch {}] dev\n{}'.format(epoch, metric_dev), hp.telegram_bot_token, hp.telegram_chat_id)
            report_to_telegram('[epoch {}] test\n{}'.format(epoch, metric_test), hp.telegram_bot_token, hp.telegram_chat_id)

        if argument_f1_test >best_f1:
            print("角色词 F1 值由 {:.3f} 更新至 {:.3f} ".format(best_f1, argument_f1_test))
            best_f1 = argument_f1_test
            print("=======保存模型=======")
            torch.save(model, hp.model_path)
            no_gain_rc = 0
        else:
            no_gain_rc = no_gain_rc+1

        ## 提前终止
        if no_gain_rc > hp.early_stop:
            print("连续{}个epoch没有提升，在epoch={}提前终止".format(no_gain_rc,epoch))
            break
