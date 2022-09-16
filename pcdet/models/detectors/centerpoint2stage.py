from .detector3d_template import Detector3DTemplate
import torch


class Centerpoint2stage(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
    
    # def post_processing(self, batch_dict):
    
    #     post_process_cfg = self.model_cfg.POST_PROCESSING
    #     batch_size = batch_dict['batch_size']
    #     recall_dict = {}
    #     pred_dicts = []
    #     for index in range(batch_size):
    #         if batch_dict.get('batch_index', None) is not None:
    #             assert batch_dict['batch_box_preds'].shape.__len__() == 2
    #             batch_mask = (batch_dict['batch_index'] == index)
    #         else:
    #             assert batch_dict['batch_box_preds'].shape.__len__() == 3
    #             batch_mask = index

    #         box_preds = batch_dict['batch_box_preds'][batch_mask]
    #         src_box_preds = box_preds

    #         if not isinstance(batch_dict['batch_cls_preds'], list):
    #             cls_preds = batch_dict['batch_cls_preds'][batch_mask]

    #             src_cls_preds = cls_preds
    #             assert cls_preds.shape[1] in [1, self.num_class]
    #             if not batch_dict['cls_preds_normalized']:
    #                 cls_preds = torch.sigmoid(cls_preds)
    #         else:
    #             cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
    #             src_cls_preds = cls_preds
    #             if not batch_dict['cls_preds_normalized']:
    #                 cls_preds = [torch.sigmoid(x) for x in cls_preds]

        
    #         scores = torch.sqrt(cls_preds.reshape(-1) * batch_dict['roi_scores'][index].reshape(-1))
        
    #         label_preds = batch_dict['roi_labels'][batch_mask]
    #         mask = (label_preds != 0).reshape(-1)

    #         box_preds = box_preds[mask, :]
    #         scores = scores[mask]
    #         labels = label_preds[mask]-1
        
    #         recall_dict = self.generate_recall_record(
    #             box_preds=box_preds if 'rois' not in batch_dict else src_box_preds,
    #             recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
    #             thresh_list=post_process_cfg.RECALL_THRESH_LIST)

    #         # currently don't need nms 
    #         pred_dict = {
    #             'pred_boxes': box_preds,
    #             'pred_scores': scores,
    #             'pred_labels': labels  
    #             }

    #         pred_dicts.append(pred_dict)
    
    #     return pred_dicts, recall_dict
