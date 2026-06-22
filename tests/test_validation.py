"""Tests for validation splitters."""

import pandas as pd

from src.validation import OOTSplitter, StratifiedKFoldCV


class TestOOTSplitter:
    """Test Out-of-Time splitter."""

    def test_no_temporal_overlap(self, sample_borrower_df):
        """Train and OOT sets should have no date overlap."""
        splitter = OOTSplitter(cutoff_date="2024-07-01")
        train_df, oot_df = splitter.split(sample_borrower_df)

        train_max = pd.to_datetime(train_df["origination_date"]).max()
        oot_min = pd.to_datetime(oot_df["origination_date"]).min()

        assert train_max < oot_min

    def test_train_dates_before_cutoff(self, sample_borrower_df):
        """All train dates should be before the cutoff."""
        splitter = OOTSplitter(cutoff_date="2024-07-01")
        train_df, _ = splitter.split(sample_borrower_df)

        train_dates = pd.to_datetime(train_df["origination_date"])
        assert (train_dates < pd.Timestamp("2024-07-01")).all()

    def test_oot_dates_after_cutoff(self, sample_borrower_df):
        """All OOT dates should be on or after the cutoff."""
        splitter = OOTSplitter(cutoff_date="2024-07-01")
        _, oot_df = splitter.split(sample_borrower_df)

        oot_dates = pd.to_datetime(oot_df["origination_date"])
        assert (oot_dates >= pd.Timestamp("2024-07-01")).all()

    def test_complete_split(self, sample_borrower_df):
        """Train + OOT should equal total records."""
        splitter = OOTSplitter(cutoff_date="2024-07-01")
        train_df, oot_df = splitter.split(sample_borrower_df)

        assert len(train_df) + len(oot_df) == len(sample_borrower_df)


class TestStratifiedKFoldCV:
    """Test stratified k-fold cross-validation."""

    def test_correct_number_of_folds(self, sample_borrower_df):
        """Should generate correct number of folds."""
        y = sample_borrower_df["default_flag"].values
        cv = StratifiedKFoldCV(n_folds=3)
        folds = cv.split(sample_borrower_df, y)

        assert len(folds) == 3

    def test_no_overlap_between_folds(self, sample_borrower_df):
        """Train and validation indices in each fold should not overlap."""
        y = sample_borrower_df["default_flag"].values
        cv = StratifiedKFoldCV(n_folds=3)
        folds = cv.split(sample_borrower_df, y)

        for train_idx, val_idx in folds:
            assert len(set(train_idx) & set(val_idx)) == 0
