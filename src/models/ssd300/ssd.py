
import tensorflow as tf
import numpy as np

from model.base_vgg16 import VGG16

class SSD(VGG16):

    def __init__(self):
        super()

    def build(self, input, is_training=True):
        """
        SSD network by use of VGG16.
        This assembles SSD network by extending VGG16 class.
        Args:
            input: images batch
            is_training: is this training?
        Returns:
            feature maps, pred_confs, pred_locs
        """

        # assenble base network
        self.base = super().build(input, is_training)
        
        self.conv6 = convolution(self.base, 'conv6')
        self.pool6 = pooling(self.conv6, 'pool6')
        self.conv7 = convolution(self.pool6, 'conv7')

        self.conv8_1 = convolution(self.conv7, 'conv8_1')
        self.conv8_2 = convolution(self.conv8_1, 'conv8_2', ksize=3, stride=2)

        self.conv9_1 = convolution(self.conv8_2, 'conv9_1')
        self.conv9_2 = convolution(self.conv9_1, 'conv9_2', ksize=3, stride=2)

        self.conv10_1 = convolution(self.conv9_2, 'conv10_1')
        self.conv10_2 = convolution(self.conv10_1, 'conv10_2', ksize=3, stride=2)

        self.conv11_1 = convolution(self.conv10_2, 'conv11_1')
        self.conv11_2 = convolution(self.conv11_1, 'conv11_2', stride=3)

        print('================== Feature Map is Below ==================')

        self.feature_maps = []
        # extra feature maps
        self.feature_maps.append(convolution(self.base, 'map1'))
        self.feature_maps.append(convolution(self.conv7, 'map2'))
        self.feature_maps.append(convolution(self.conv8_2, 'map3'))
        self.feature_maps.append(convolution(self.conv9_2, 'map4'))
        self.feature_maps.append(convolution(self.conv10_2, 'map5'))
        self.feature_maps.append(convolution(self.conv11_2, 'map6'))

        pred = []
        for i, fmap in zip(range(len(self.feature_maps)), self.feature_maps):
            output_size = fmap.get_shape().as_list()
            height = output_size[1]
            width = output_size[2]
            pred.append(tf.reshape(fmap, [-1, width*height*boxes[i], classes+4]))

        concatenated = tf.concat(pred, axis=1)
        
        self.pred_confs = concatenated[:,:,:classes]
        self.pred_locs = concatenated[:,:,classes:]

        print('concatenated: '+str(concatenated))
        print('confs: '+str(self.pred_confs.get_shape().as_list()))
        print('locs: '+str(self.pred_locs.get_shape().as_list()))

        return self.feature_maps, self.pred_confs, self.pred_locs


    def smooth_L1(self, x):
        """
        smooth L1 loss func
        Args: the list of x range
        Returns: result tensor
        """
        sml1 = tf.multiply(0.5, tf.pow(x, 2.0))
        sml2 = tf.subtract(tf.abs(x), 0.5)
        cond = tf.less(tf.abs(x), 1.0)

        return tf.where(cond, sml1, sml2)


    def loss(self, total_boxes):
        """
        loss func defined as Loss = (Loss_conf + a*Loss_loc) / N
        In here, compute confidence loss and location loss,
        finally, total loss.
        Args:
            total_boxes: total size of boxes => len(self.default_boxes)
        Returns:
            total loss per batch
            confidence loss
            location loss
            positive list
            negative list
            ground truth label placeholder
            ground truth boxes placeholder
        """

        gt_labels = tf.placeholder(shape=[None, total_boxes], dtype=tf.int32)
        gt_boxes = tf.placeholder(shape=[None, total_boxes, 4], dtype=tf.float32)
        pos = tf.placeholder(shape=[None, total_boxes], dtype=tf.float32)
        neg = tf.placeholder(shape=[None, total_boxes], dtype=tf.float32)

        loss_loc = tf.reduce_sum(self.smooth_L1(gt_boxes - self.pred_locs), reduction_indices=2) * pos
        loss_loc = tf.reduce_sum(loss_loc, reduction_indices=1) / (1e-5 + tf.reduce_sum(pos, reduction_indices = 1))

        loss_conf = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_confs, labels=gt_labels) * (pos + neg)
        loss_conf = tf.reduce_sum(loss_conf, reduction_indices=1) / (1e-5 + tf.reduce_sum((pos + neg), reduction_indices = 1))

        loss = tf.reduce_sum(loss_conf + loss_loc)

        return loss, loss_conf, loss_loc, pos, neg, gt_labels, gt_boxes


    def _filter(self, pconfs, plocs):
        """
        exclude extra boxes.
        Args:
            pconfs, plocs:
                confidences and locations whose confidence is in top 200 and filtered by threshold min 0.1.
        Returns:
            filtered locations and its labels
        """

        jacc_th = 0.3
        det_locs = []
        det_labels = []

        def jacc_filter(plabel, ploc):
            for dlabel, dloc in zip(det_labels, det_locs):
                jacc = jaccard(center2corner(dloc), center2corner(ploc))

                # meaning this is same object
                if dlabel == plabel and jacc_th < jacc:
                    return False
            return True

        for pconf, ploc in zip(pconfs, plocs):
            plabel = np.argmax(pconf)

            if plabel != classes-1 and jacc_filter(plabel, ploc):
                det_locs.append(ploc)
                det_labels.append(plabel)

        return det_locs, det_labels

    
    def non_maximum_suppression(self, candidates, overlap_threshold):
        """
        this is nms(non maximum_suppression) which filters predicted objects.
        Args:
            predicted bounding boxes
        Returns:
            detected bounding boxes and its label
        """

        label = candidates[:,4]
        boxes = candidates[label<classes-1]

        if len(boxes) == 0:
            return []

        picked = []

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2] + x1
        y2 = boxes[:,3] + y1

        area = (boxes[:,2]) * (boxes[:,3])
        idxs = np.argsort(x1)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            picked.append(i)
            suppress = [last]
    
            for pos in range(0, last):
                j = idxs[pos]
    
                # extract smallest and largest bouding boxes
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
    
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)

                # overlap of current box and those in area list
                overlap = float(w * h) / area[j]
    
                # suppress current box
                if overlap > overlap_threshold:
                    suppress.append(pos)
    
            # delete suppressed indexes
            idxs = np.delete(idxs, suppress)
    
        return boxes[picked]
        

    def detect_objects(self, pred_confs, pred_locs):
        """
        this method returns detected objects list (means high confidences locs and its labels)
        Args is computed Tensor.
        Args:
            pred_confs: predicated confidences ( output of matching() )
            pred_locs: predicated locations ( output of matching() )
        Returns:
            detected locs and its labels
        """

        detected_locs = []
        detected_labels = []
        hist = [0 for _ in range(classes)]
        for conf, loc in zip(pred_confs[0], pred_locs[0]):
            hist[np.argmax(conf)] += 1
        # print(hist)

        # extract top 200 by confidence
        possibilities = [np.amax(np.exp(conf)) / (np.sum(np.exp(conf)) + 1e-3) for conf in pred_confs[0]]
        indicies = np.argpartition(possibilities, -200)[-200:]
        top200 = np.asarray(possibilities)[indicies]
        slicer = indicies[0.9 < top200]
        locations, labels = self._filter(pred_confs[0][slicer], pred_locs[0][slicer])

        locations, labels = pred_locs[0][slicer], np.argmax(pred_confs[0][slicer], axis=1)
        labels = np.asarray(labels).reshape(len(labels), 1)
        with_labels = np.concatenate((locations, labels), axis=1)
        
        # labels, locations = image.non_max_suppression(boxes, possibilities, 10)
        filtered = self.non_maximum_suppression(with_labels, 0.1)
        # locations, labels = pred_confs[0][indices], pred_locs[0][indices]
        if len(filtered) == 0:
            filtered = np.zeros((4, 5))

        return filtered[:,:4], filtered[:,4]
        #return locations, labels