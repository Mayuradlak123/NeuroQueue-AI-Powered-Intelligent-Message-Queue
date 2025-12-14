import torch
import re

class FeatureEngineer:
    """
    Extracts statistical features to augment semantic embeddings.
    """
    def __init__(self):
        self.num_features = 10 # 384 (embedding) + 10 (stats) = 394 total dim

    def extract_features(self, text: str) -> list:
        """
        Returns a list of 10 statistical features.
        """
        features = []
        
        length = len(text)
        words = text.split()
        num_words = len(words)
        
        # 1. Normalized Length
        features.append(min(length / 500.0, 1.0))
        
        # 2. Word count norm
        features.append(min(num_words / 100.0, 1.0))
        
        # 3. Capitalization Ratio (Urgency)
        caps_count = sum(1 for c in text if c.isupper())
        features.append(caps_count / max(length, 1))
        
        # 4. Digit Ratio (Ids, values)
        digit_count = sum(1 for c in text if c.isdigit())
        features.append(digit_count / max(length, 1))
        
        # 5. Exclamation Count
        features.append(min(text.count("!") / 5.0, 1.0))
        
        # 6. Question Mark Count
        features.append(min(text.count("?") / 5.0, 1.0))
        
        # 7. Special Char Ratio (Code snippets?)
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        features.append(special_chars / max(length, 1))
        
        # 8. "Error" / "Fail" keyword presence
        has_error = 1.0 if re.search(r'\b(error|fail|exception|crash)\b', text.lower()) else 0.0
        features.append(has_error)
        
        # 9. "Payment" / "Money" keyword presence
        has_payment = 1.0 if re.search(r'\b(pay|card|invoice|transaction)\b', text.lower()) else 0.0
        features.append(has_payment)
        
        # 10. Placeholder
        features.append(0.0)
        
        return features
