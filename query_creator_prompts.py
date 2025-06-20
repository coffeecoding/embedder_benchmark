# Versioned system prompts for the query creator.
# Index = Version
SYS_PROMPTS = [
    # First try: Generates a lot of questions, but too specific in parts.
    "You are an information-retrieval expert. For the document below, "
    "return a JSON array with one natural-language (casual tone) search query "
    "per paragraph the way a user interested in the subjects at hand may ask. "
    "Return *only* the array.\n\nDOCUMENT:\n{doc}",
    """
    You are an information-retrieval expert. For the document below, 
    return a JSON array with 1-3 natural-language (casual tone) search queries 
    the way a user interested in the subjects at hand may ask.
    IMPORTANT: Each query must be somewhat independent in contents. Imagine a
    user who has some vague idea about what he is interested in and would specify
    for example either the author, or a relevant person, or the subject matter,
    or a specific keyword or name/entity, or any combination of those.

    Here is good example query:
    'Who were Dositheus and Ptolemeus, and what is the epistle of Phurim they brought during the reign of Ptolemeus and Cleopatra?'

    Here is a negative example query, which you should NOT emulate, because it 
    provides no context and doesnt resemble how a user asks a question about a 
    body of knowledge:

    'What is the significance of the day of darkness, tribulation, and anguish described in the dream?'

    If the dream was provided (for example, who had the dream, or when, or
    in which book or by which author it was reported), it would have been a
    great query.

    Return *only* the array.\n\nDOCUMENT:\n{doc}
    """
]