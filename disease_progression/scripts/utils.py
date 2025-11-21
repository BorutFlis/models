import re


def extract_stratify_HF_abbr(attr_name: str) -> str:
    """
    matching the names from disease progression dataset, the filters have increasing inclusivity

    """
    if first_occurence_match := re.search("^summary_first_occurence_(.+)", attr_name):
        return first_occurence_match.groups()[0]
    if med_match := re.search("summary_(.+)_ever_taken", attr_name):
        return med_match.groups()[0]
    if summary_match := re.search("summary_([A-Za-z]{3}_[A-Za-z]{2,5})", attr_name):
        return summary_match.groups()[0]
    return attr_name
