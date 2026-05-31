from symspellpy import SymSpell

def convert_text_to_corrected(original_text):
    
    # Initialize SymSpell
    sym_spell = SymSpell(max_dictionary_edit_distance=2)

    # Load the frequency dictionary
    dictionary_path = r"D:\Translate\Translator\Models\frequency_dictionary_en_82_765.txt"
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    # Correct the text
    suggestions = sym_spell.lookup_compound(original_text, max_edit_distance=2)
    return suggestions[0].term if suggestions else original_text
