
columns = ["action","record_type","libid","serial_title","nlm_unique_id","holdings_format","begin_volume","end_volume","issns","currently_received","retention_policy","limited_retention_period","limited_retention_type","embargo_period","has_epub_ahead_of_print","has_supplements","ignore_warnings","last_modified","begin_year","end_year"]
columns.insert(8, columns.pop(18))
columns.insert(9, columns.pop(19))
print(columns)
