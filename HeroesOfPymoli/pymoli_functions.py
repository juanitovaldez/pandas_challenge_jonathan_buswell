#Creates a summary from a data_frame with field_names as a list of cols
#<# Unique Items, Avg Price, # Purchases, Total Revenue
#Features to add: data size check ie fail if
def purchase_summary(data_frame, field_names)
    unique_items = data_frame