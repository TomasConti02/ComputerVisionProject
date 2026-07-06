
class RealTimeOODDetector:
    def __init__( self,  centroids,  window_size=400,  batch_size=40,  initial_threshold=None,  smoothing=0.9,  min_samples=50,  
                 percentile=90,  max_percentile=99,  safe_threshold=None,  max_consecutive_drops=5, max_consecutive_ups=5,
                  median_window=5, history_buffer_size=20 , smooth_safety_th=False):
        # idk if use distance_buffer or batch_distances for percentile operation
        self.knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        self.knn.fit(centroids)
        self.distance_buffer = deque(maxlen=window_size)  
        #self.batch_distances = []   # if we don't use this batch_distances for percentile operation it can be replace with a counter  
        self.batch_distances = 0                 
        self.threshold = initial_threshold
        self.safe_threshold = safe_threshold
        self.batch_size = batch_size 
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.percentile = percentile
        self.max_percentile = max_percentile
        self.max_consecutive_drops = max_consecutive_drops
        self.max_consecutive_ups = max_consecutive_ups
        self.consecutive_drops = 0
        self.consecutive_ups = 0
        self.median_window = median_window
        self.batch_percentiles = deque(maxlen=median_window)
        self.smooth_safety_th = smooth_safety_th

        if safe_threshold is None and smooth_safety_th is True:
            raise ValueError("Incompatible parameters: smooth_safety_th cannot be True if safe_threshold is None")

        self.threshold_flow = deque(maxlen=max(max_consecutive_drops, max_consecutive_ups)+1)
        self.threshold_trend = []
        self.counter = 0
        self.history = {   "time": [],   "distances": [],    "thresholds": [],     "predictions": [],   "ground_truth": [],   "trend": []  }

    def _compute_raw_threshold(self, distances):
        if len(distances) < self.min_samples:
            return None
        #return np.percentile(np.unique(distances), self.percentile)  
        #return np.percentile(distances, self.percentile) # real time operation 
        #return np.percentile(np.unique(self.distance_buffer), self.percentile) # delete duplicates before percentile, equals weights IDK 
        return np.percentile(self.distance_buffer, self.percentile)

    def _fallback_threshold(self, distances): #fall back policy
        median = np.median(distances)
        mad = np.median(np.abs(distances - median))
        return median + 2.0 * mad

    def _update_threshold(self):
        if len(self.distance_buffer) < self.min_samples:
            return
        buffer = np.asarray(self.distance_buffer)
        thr = self._compute_raw_threshold(buffer) #calculate the th percentile 
        if thr is None:
            thr = self._fallback_threshold(buffer)
        self.batch_percentiles.append(thr) #history of th 
        if len(self.batch_percentiles) >= 2:
            thr_median = np.median(self.batch_percentiles) #cloud be helpful in case of the outliers/spike
        else:
            thr_median = thr
        thr_candidate = min(thr_median, np.percentile(buffer, self.max_percentile)) # another security check con the th values
        if self.threshold is None:
            new_thr = thr_candidate
            trend = "INIT"
        else:
            if thr_candidate > self.threshold:
                self.consecutive_ups += 1 #update ups 
                self.consecutive_drops = 0 #resent drops
                trend = "UP"
            elif thr_candidate < self.threshold:
                self.consecutive_drops += 1 #update drops
                self.consecutive_ups = 0 #reset ups
                trend = "DOWN" 
            else: #reset all because it is stable
                self.consecutive_ups = 0
                self.consecutive_drops = 0
                trend = "SAME"

            smoothing_target = thr_candidate 
            is_reset = False
            
            if self.safe_threshold is not None and (self.consecutive_drops >= self.max_consecutive_drops or self.consecutive_ups >= self.max_consecutive_ups): # case with self.safe_threshold
                smoothing_target = self.safe_threshold
                trend = "RESET_SAFE"
                self.consecutive_drops = 0
                self.consecutive_ups = 0
                if not self.smooth_safety_th: #if there is no smooth_safety_th execute a hard cut of the th value, if it is not the smoothing_target will be smoothed
                    is_reset=True

            elif self.consecutive_drops >= self.max_consecutive_drops and self.safe_threshold is None: 
                back_index = self.consecutive_drops + 1 
                if len(self.threshold_flow) >= back_index:
                    smoothing_target = self.threshold_flow[-back_index] #back track
                    trend = "RESET_DOWN"
                else:
                    trend = "SAME"
                self.consecutive_drops = 0
                self.consecutive_ups = 0

            elif self.consecutive_ups >= self.max_consecutive_ups and self.safe_threshold is None:
                back_index = self.consecutive_ups + 1 
                if len(self.threshold_flow) >= back_index:
                    smoothing_target = self.threshold_flow[-back_index] #back track
                    trend = "RESET_UP"
                else:
                    trend = "SAME"
                self.consecutive_ups = 0
                self.consecutive_drops = 0

            if is_reset:
              new_thr = smoothing_target
            else:
              new_thr = self.smoothing * self.threshold + (1 - self.smoothing) * smoothing_target #allways

        self.threshold = new_thr # the current th value became the old one
        #self.threshold_flow.append(new_thr) #history of the past th values for backtrack
        self.threshold_flow.append(thr_candidate) # BETTER !!!!!!
        self.threshold_trend.append(trend)

    def process(self, embedding, true_label=None):

        dist = self.knn.kneighbors(np.asarray(embedding).reshape(1, -1))[0][0][0] #distance inference
        self.distance_buffer.append(dist) #update hisotry distance buffer 
        #self.batch_distances.append(dist) 
        self.batch_distances+=1
        self.counter += 1

        #if len(self.batch_distances) >= self.batch_size:
        if self.batch_distances >= self.batch_size:
            #self.batch_distances.clear()
            self.batch_distances = 0
            self._update_threshold() #after a batch of distance update the th

        if self.threshold is None and len(self.distance_buffer) >= self.min_samples: #end up of the warm up 
            buffer = np.asarray(self.distance_buffer)
            thr = self._compute_raw_threshold(buffer) 
            if thr is None:
                thr = self._fallback_threshold(buffer) 
            self.threshold = min(thr, np.percentile(buffer, self.max_percentile)) # now threshold is not more None

        is_ood = self.threshold is not None and dist > self.threshold # classification operation

        trend = self.threshold_trend[-1] if self.threshold_trend else "INIT"

        self.history["time"].append(self.counter - 1)
        self.history["distances"].append(dist)
        self.history["thresholds"].append(self.threshold if self.threshold is not None else np.nan)
        self.history["predictions"].append(int(is_ood))
        self.history["ground_truth"].append(0 if true_label is None else true_label)
        self.history["trend"].append(trend)

        return dist, is_ood, self.threshold
