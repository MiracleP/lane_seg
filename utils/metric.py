import numpy as np

def compute_iou(pred, gt, result):
    '''
    :param pred: shape[N, H, W], predict label from model
    :param gt:  shape[N, H, W], ground true
    :param result: 保存交并数据
    :return: result: tp, ta
    '''
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    for i in range(8):
        single_pred = pred == i
        single_gt = gt == i
        temp_tp = np.sum(single_pred * single_gt)
        temp_ta = np.sum(single_gt) + np.sum(single_pred) - temp_tp
        result['TP'][i] += temp_tp
        result['TA'][i] += temp_ta
    return result

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    label = label.view(-1, size[-2], size[-1])
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix