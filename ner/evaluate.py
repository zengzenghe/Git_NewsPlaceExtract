from .models.crf import CRFModel
from .utils import save_model, flatten_lists
from .evaluating import Metrics
import pickle


def crf_train_eval(train_data, test_data, remove_O=False):
    # 训练CRF模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, "./ckpts/crf.pkl")

    pred_tag_lists = crf_model.test(test_word_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def crf_test(test_data, remove_O=False):
    test_word_lists, test_tag_lists = test_data

    # crf_model = CRFModel()
    f = open("./ckpts/crf.pkl", 'rb')
    crf_model = pickle.load(f)
    f.close()
    # input type is list
    pred_tag_lists = crf_model.test(test_word_lists)
    pred_tag_lists2 = crf_model.test(test_word_lists[0:1])

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists
