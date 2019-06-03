from ner.data import build_corpus
from ner.evaluate import crf_train_eval

from extractPlace import FileTools
from datetime import datetime
from ner.utils import load_model


def crf_predict_news(model, lst):
    pred_tags = model.test(lst)
    return pred_tags


def save_predict_news(tokens_lst, labels_lst, index):
    news_labels = []
    news_tokens = []
    i = 0
    for tokens, labels in zip(tokens_lst, labels_lst):
        # print(i)
        assert len(tokens) == len(labels)
        news_labels.append('[CLS]')
        news_tokens.append('[CLS]')
        news_labels.extend(labels)
        news_tokens.extend(tokens)
        news_labels.append('[SEP]')
        news_tokens.append('[SEP]')
        i += 1
    tokens_path = '../output/predict_dir/news_tokens/token_test_' + str(index) + '.txt'
    labels_path = '../output/predict_dir/news_labels/label_test_' + str(index) + '.txt'
    with open(tokens_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(news_tokens))

    with open(labels_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(news_labels))


def train():
    print("训练模型,评估结果!")
    print("读取数据...")
    corpus_path = 'corpus'
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", make_vocab=True, data_dir=corpus_path)
    # dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 训练评估CRF模型
    print("正在训练评估CRF模型...")
    crf_train_eval((train_word_lists, train_tag_lists), (test_word_lists, test_tag_lists)
                   )


def test():
    input_file = '../data/1000_news.xls'
    input_file = '../data/1050_news.xls'
    input_file = '../data/3150_news.xls'
    df = FileTools.read_xls_data(input_file)
    news_list = []
    for index, row in df.iterrows():
        lst = []
        for v in row['text']:
            lst.append(v[0])
        news_list.append(lst)

    crf_model = load_model('./ckpts/crf.pkl')
    start_time = datetime.now()
    for i in range(len(news_list)):
        # if i != 999:
        #     continue
        print('index:', i)
        crf_pred = crf_predict_news(crf_model, news_list[i])
        save_predict_news(news_list[i], crf_pred, i)
    end_time = datetime.now()
    print('time', end_time - start_time)


def main():
    """训练模型，评估结果"""
    # train()
    """test news"""
    test()


if __name__ == "__main__":
    main()
