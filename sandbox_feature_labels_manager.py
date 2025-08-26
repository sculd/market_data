import copy
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

    print(labels_manager.load("ffd_zscore"))
    
    feature_collection = FeatureLabelCollection()
    feature_labels = market_data.feature.registry.list_registered_features('all')
    print(f"Found {len(feature_labels)} registered features: {feature_labels}")
    

    # generate all possible combinations of collections
    feature_label_objs = [FeatureLabel(feature_label) for feature_label in feature_labels]        
    feature_collection_list = FeatureLabelCollectionsManager.get_super_set_collections(feature_label_objs)

    for feature_collection in feature_collection_list:
        tag = FeatureLabelCollectionsManager.get_tag(feature_collection)
        if not tag:
            continue
        labels_manager.save(feature_collection, tag)
        print(f"Saved collection with {len(feature_collection.feature_labels)} features {tag[:10]}...")

    feature_collection = FeatureLabelCollection()

    for feature_label in feature_labels:
        feature_collection = feature_collection.with_feature_label(FeatureLabel(feature_label))

    # Save the collection with all features
    labels_manager.save(feature_collection, "all_features")
    print(f"Saved collection with all {len(feature_collection.feature_labels)} features")
    
    # List all available tags
    print(f"All available tags: {labels_manager.list_tags()}")

