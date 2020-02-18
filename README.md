# Transformer-based-pretrained-model-for-event-extraction
使用bert/gpt2/albert/xlm/roberta等预训练语言模型在ace2005上进行事件抽取任务。 代码在nlpcl-lab / bert-event-extraction框架上修改，使用pytorch 的transformer和crf模型替换了原项目的模型构建部分。 模型整体采用序列标注的方式，未使用任何辅助信息。 先用crf做触发词识别，再根据触发词识别结果再用crf进行论元识别，预训练模型选用xlm-roberta-large时，trigger-f1=0.72; argument-f1=0.45。argument提升了0.05。

[trigger classification]	P=0.677	R=0.754	F1=0.713
[argument classification]	P=0.588	R=0.384	F1=0.464
[trigger identification]	P=0.723	R=0.805	F1=0.762
[argument identification]	P=0.617	R=0.403	F1=0.488

超参如下
==================== 超参 ====================
        PreTrainModel = ['Bert_large', 'Gpt', 'Gpt2', 'Ctrl', 'TransfoXL', 'Xlnet_base', 'Xlnet_large', 'XLM', 'DistilBert_base', 'DistilBert_large', 'Roberta_base', 'Roberta_large', 'XLMRoberta_base', 'XLMRoberta_large', 'ALBERT-base-v1', 'ALBERT-large-v1', 'ALBERT-xlarge-v1', 'ALBERT-xxlarge-v1', 'ALBERT-base-v2', 'ALBERT-large-v2', 'ALBERT-xlarge-v2', 'ALBERT-xxlarge-v2']
           early_stop = 5
                   lr = 1e-05
                   l2 = 1e-05
             n_epochs = 50
               logdir = logdir
             trainset = data/train_balance.json
               devset = data/dev.json
              testset = data/test.json
           LOSS_alpha = 1.0
   telegram_bot_token = 
     telegram_chat_id = 
       PreTrain_Model = XLMRoberta_large
           model_path = /content/drive/My Drive/Colab Notebooks/模型/事件抽取/Transformer-based-pretrained-model-for-event-extraction-master/save_model/latest_model.pt
           batch_size = 16
