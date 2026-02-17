import numpy as np
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.models import LinearHarmonicModel
from src import config
import os
import joblib

class EnsembleHarmonicModel:
    """
    Ensemble model combining LinearHarmonicModel (Recall) and XGBoost (Precision)
    via a Meta-Learner (Decision Tree / Logistic Regression).
    """
    def __init__(self, device='cpu', xgb_params=None):
        self.device = device
        
        # Default XGB Params
        default_xgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42,
            'eval_metric': 'logloss'
        }

        if xgb_params:
            default_xgb_params.update(xgb_params)

        # Level 0 Models
        self.linear_model = LinearHarmonicModel().to(self.device)
        self.xgb_model = xgb.XGBClassifier(**default_xgb_params)
        
        # Level 1 Meta-Learner
        # Simple Decision Tree to learn non-linear decision boundary from probs
        # or Logistic Regression for weighted voting.
        self.meta_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        
    def fit(self, train_loader, val_loader=None, pos_weight=1.0, lr_linear=None, epochs_linear=None):
        """
        Trains all components of the ensemble.
        """
        # 1. Train Linear Model
        print("Training Linear Model...")

        # Use config defaults if not provided
        lr = lr_linear if lr_linear else config.LEARNING_RATE
        epochs = epochs_linear if epochs_linear else config.EPOCHS

        self._train_linear_model(train_loader, pos_weight, lr, epochs)
        
        # 2. Train XGBoost
        print("Training XGBoost...")
        X_xgb, y_xgb = self._extract_features(train_loader, 'classifier_features')
        self.xgb_model.scale_pos_weight = pos_weight
        self.xgb_model.fit(X_xgb, y_xgb)
        
        # 3. Generate Meta-Features via Cross-Validation (to avoid overfitting)
        # Or validation set if provided.
        print("Generating Meta-Features...")
        
        if val_loader:
            meta_X, meta_y = self._generate_meta_features(val_loader)
        else:
            # Fallback: Generate predictions on training set (riskier but simple)
            meta_X, meta_y = self._generate_meta_features(train_loader)
        
        # 4. Train Meta-Learner
        print("Training Meta-Learner...")
        if len(meta_X) > 0:
            self.meta_model.fit(meta_X, meta_y)
        else:
            print("Warning: No meta-features generated.")

        print("Ensemble Training Complete.")

    def predict_proba(self, lin_feat, xgb_feat):
        """
        Returns the probability from the meta-learner.
        lin_feat: Tensor (1, 20)
        xgb_feat: Numpy (1, 44)
        """
        # Level 0 Predictions
        self.linear_model.eval()
        with torch.no_grad():
            lin_input = torch.tensor(lin_feat, dtype=torch.float).to(self.device)
            if lin_input.dim() == 1: lin_input = lin_input.unsqueeze(0)
            p_lin = torch.sigmoid(self.linear_model(lin_input)).item()
            
        # Ensure xgb_feat is 2D
        if xgb_feat.ndim == 1:
            xgb_feat = xgb_feat.reshape(1, -1)
            
        p_xgb = self.xgb_model.predict_proba(xgb_feat)[0][1]
        
        # Level 1 Prediction
        meta_input = np.array([[p_lin, p_xgb]])
        p_meta = self.meta_model.predict_proba(meta_input)[0][1]
        
        return p_lin, p_xgb, p_meta

    def _train_linear_model(self, loader, pos_weight, lr, epochs):
        optimizer = torch.optim.Adam(self.linear_model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        
        self.linear_model.train()
        for epoch in range(epochs):
            for batch in loader:
                batch = batch.to(self.device)
                labels = batch.y.unsqueeze(1)
                lin_feat = batch.linear_features
                if lin_feat.dim() == 3: lin_feat = lin_feat.squeeze(1)
                
                optimizer.zero_grad()
                out = self.linear_model(lin_feat)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

    def _extract_features(self, loader, feature_key):
        X_list = []
        y_list = []
        for batch in loader:
            feats = getattr(batch, feature_key)
            if feats.dim() == 3: feats = feats.squeeze(1)
            X_list.append(feats.cpu().numpy())
            y_list.append(batch.y.cpu().numpy())
            
        if not X_list:
            return np.array([]), np.array([])
        return np.vstack(X_list), np.hstack(y_list)

    def _generate_meta_features(self, loader):
        meta_X = []
        meta_y = []
        
        self.linear_model.eval()
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Linear Prob
                lin_feat = batch.linear_features
                if lin_feat.dim() == 3: lin_feat = lin_feat.squeeze(1)
                out_lin = torch.sigmoid(self.linear_model(lin_feat)).cpu().numpy().flatten()
                
                # XGB Prob
                xgb_feat = batch.classifier_features
                if xgb_feat.dim() == 3: xgb_feat = xgb_feat.squeeze(1)
                xgb_feat_np = xgb_feat.cpu().numpy()
                
                # Predict XGB
                if len(xgb_feat_np) > 0:
                    out_xgb = self.xgb_model.predict_proba(xgb_feat_np)[:, 1]
                else:
                    out_xgb = np.array([])

                if len(out_lin) == len(out_xgb):
                    batch_meta = np.column_stack((out_lin, out_xgb))
                    meta_X.append(batch_meta)
                    meta_y.append(batch.y.cpu().numpy())
                
        if not meta_X:
            return np.array([]), np.array([])
        return np.vstack(meta_X), np.hstack(meta_y)
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.save(self.linear_model.state_dict(), os.path.join(path, 'linear.pth'))
        joblib.dump(self.xgb_model, os.path.join(path, 'xgb.pkl'))
        joblib.dump(self.meta_model, os.path.join(path, 'meta.pkl'))
        
    def load(self, path):
        self.linear_model.load_state_dict(torch.load(os.path.join(path, 'linear.pth'), map_location=self.device))
        self.xgb_model = joblib.load(os.path.join(path, 'xgb.pkl'))
        self.meta_model = joblib.load(os.path.join(path, 'meta.pkl'))
        self.linear_model.eval()
