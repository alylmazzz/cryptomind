# testler ölçülmüş ek evren dosyasından etkilenmesin (kod-tanımlı 40 parite)
import os
os.environ.setdefault("CRYPTOMIND_EXTRA_SYMBOLS", "0")
