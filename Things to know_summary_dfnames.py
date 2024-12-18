# Data Summary Storage Class
class DataSummary:
    """Class to store and manage analysis results"""
    def __init__(self):
        self.summaries = {}
    
    def add_summary(self, dataset_name, summary_type, data):
        """Add summary statistics to storage"""
        if dataset_name not in self.summaries:
            self.summaries[dataset_name] = {}
        self.summaries[dataset_name][summary_type] = data
    
    def get_summary(self, dataset_name, summary_type=None):
        """Retrieve stored summary statistics"""
        if summary_type:
            return self.summaries.get(dataset_name, {}).get(summary_type)
        return self.summaries.get(dataset_name)
    
    def print_summary(self, dataset_name):
        """Print stored summaries for a dataset"""
        if dataset_name in self.summaries:
            print(f"\nSummary for {dataset_name}:")
            for summary_type, data in self.summaries[dataset_name].items():
                print(f"\n{summary_type}:")
                print(data)

# Initialize summary storage
summary = DataSummary()


# Clean and prepare the data
shifts_df = load_and_clean_shifts(shifts_df)
bookings_df = load_and_clean_bookings(bookings_df)
cancellations_df = load_and_clean_cancellations(cancellations_df)
