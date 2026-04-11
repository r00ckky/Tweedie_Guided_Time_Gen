import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import QuantileTransformer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class AmexDataset(Dataset):
    def __init__(self, customer_df, db_path, fill_dict=None, transformer=None, max_seq_len=13):
        self.customer_df = customer_df
        self.db_path = db_path
        self.fill_dict = fill_dict if fill_dict is not None else {}
        self.max_seq_len = max_seq_len
        
        self.transformer = transformer if transformer is not None else QuantileTransformer(output_distribution='normal', random_state=42)
        
        self.cols_to_drop = [
            'D_87', 'D_88', 'D_108', 'D_110', 'D_111', 'B_39', 'D_73', 'B_42', 
            'D_135', 'D_134', 'D_137', 'D_138', 'D_136', 'R_9', 'B_29', 'D_106', 
            'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142', 'D_53', 
            'D_82', 'D_50', 'B_17', 'D_105', 'D_56', 'S_9', 'B_30', 'B_38', 
            'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68', 'D_77'
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("PRAGMA table_info(statements)")
        all_cols = [col[1] for col in cursor.fetchall()]
        self.good_cols = [col for col in all_cols if col not in self.cols_to_drop]
        self.select_string = ", ".join(self.good_cols)
        conn.close()
        
        self.conn = None 
        
        if not self.fill_dict:
            self.fit_transform()

    def fit_transform(self):
        print("Pulling sample data to fit the Transformer...")
        customer_samples = self.customer_df.sample(n=1000, random_state=42)['customer_ID'].tolist()
        
        conn = sqlite3.connect(self.db_path)
        all_data = []
        for customer in customer_samples:
            query = f"SELECT {self.select_string} FROM statements WHERE customer_ID = '{customer}'"
            df = pd.read_sql(query, conn)
            all_data.append(df)
        conn.close()
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            sample_df = pd.concat(all_data, ignore_index=True)
        
        sample_features = sample_df.drop(columns=['customer_ID', 'S_2'], errors='ignore')
        
        sample_features = sample_features.replace([np.inf, -np.inf], np.nan)
        
        print("Calculating medians...")
        medians = sample_features.median(numeric_only=True)
        
        self.fill_dict = medians.fillna(0).to_dict()
        
        sample_filled = sample_features.fillna(self.fill_dict)
        sample_filled = sample_filled.fillna(0) 
        
        print("Fitting Quantile Transformer (This will be instant)...")
        self.transformer.fit(sample_filled)
        print("Fit complete!")
        
        return self.fill_dict, self.transformer

    def __len__(self):
        return len(self.customer_df)

    def __getitem__(self, idx):
        # Lazy initialization: creates ONE connection per worker process
        if not hasattr(self, 'conn') or self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        
        cust_target = self.customer_df.iloc[idx]
        customer = cust_target['customer_ID']
        target = cust_target['target']
        
        query = f"SELECT {self.select_string} FROM statements WHERE customer_ID = '{customer}'"
        
        try:
            df = pd.read_sql(query, self.conn)
        except Exception as e:
            print(f"Error reading customer {customer}: {e}")
            # Return zeros on error
            num_features = len(self.good_cols) - 2
            transformed_data = np.zeros((self.max_seq_len, num_features))
            cumulative_days = np.zeros(self.max_seq_len)
        else:
            # Handle empty result
            if df.empty:
                num_features = len(self.good_cols) - 2
                transformed_data = np.zeros((1, num_features))
                cumulative_days = np.array([0.0])
            else:
                dates = pd.to_datetime(df['S_2'])
                cumulative_days = (dates - dates.iloc[0]).dt.days.values
                
                df = df.drop(columns=['customer_ID', 'S_2'], errors='ignore')
                df = df.replace([np.inf, -np.inf], np.nan)
                
                df = df.fillna(self.fill_dict).infer_objects(copy=False)
                df = df.fillna(0)
                
                transformed_data = self.transformer.transform(df)
            
            seq_len, num_features = transformed_data.shape
            
            if seq_len < self.max_seq_len:
                feature_padding = np.zeros((self.max_seq_len - seq_len, num_features))
                transformed_data = np.vstack([transformed_data, feature_padding])
                
                time_padding = np.zeros(self.max_seq_len - seq_len)
                cumulative_days = np.concatenate([cumulative_days, time_padding])
        
        X_tensor = torch.tensor(transformed_data[:self.max_seq_len], dtype=torch.float32)
        
        time_tensor = torch.tensor(cumulative_days[:self.max_seq_len] / 365.0, dtype=torch.float32).unsqueeze(1) 
        
        y_tensor = torch.tensor(float(target), dtype=torch.float32)
        
        if torch.isnan(X_tensor).any():
            print(f"CRITICAL WARNING: NaNs still detected in final tensor for customer {customer}!")
            X_tensor = torch.nan_to_num(X_tensor, nan=0.0) 
            
        return X_tensor, time_tensor, y_tensor