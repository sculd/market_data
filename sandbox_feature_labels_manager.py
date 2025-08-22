import logging
import sys
import os, datetime, pprint, itertools
import setup_env # needed for the environment variables

import market_data.feature.registry
from market_data.feature.label import FeatureLabel, FeatureLabelCollection, FeatureLabelCollectionsManager


if __name__ == '__main__':
    labels_manager = FeatureLabelCollectionsManager()
    labels_manager.save(FeatureLabelCollection(), "empty")
    labels_manager.save(FeatureLabelCollection().with_feature_label(FeatureLabel("returns")), "returns")
    labels_manager.save(
        FeatureLabelCollection().\
            with_feature_label(FeatureLabel("returns")).\
                with_feature_label(FeatureLabel("time_of_day")), "returns_with_time_of_day")
    labels_manager.save(FeatureLabelCollection().with_feature_label(FeatureLabel("ffd_zscore")), "ffd_zscore")
    labels_manager.save(FeatureLabelCollection().with_feature_label(FeatureLabel("bollinger")), "bollinger")
    labels_manager.save(FeatureLabelCollection().with_feature_label(FeatureLabel("indicators")), "indicators")
    
    print(labels_manager.list_tags())

    print(labels_manager.load("returns_with_time_of_day"))
    
    feature_collection = FeatureLabelCollection()
    feature_labels = market_data.feature.registry.list_registered_features('all')
    feature_collection = FeatureLabelCollection()

    for feature_label in feature_labels:
        feature_collection = feature_collection.with_feature_label(FeatureLabel(feature_label))

