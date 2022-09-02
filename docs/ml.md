---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3.9.12 ('ai')
  language: python
  name: python3
---

```{eval-rst}
:orphan:
```

# Μηχανική Μάθηση

Χωρίς να μπούμε σε πολλές λεπτομέρειες γύρω από τον ορισμό της **Μηχανικής Μάθησης** (*Machine Learning* -- ML), μπορούμε να πούμε ότι η ML είναι η μελέτη και σχεδίαση λογισμικού που χρησιμοποιεί την εμπειρία του παρελθόντος για τη λήψη μελλοντικών αποφάσεων. Είναι η μελέτη προγραμμάτων που μαθαίνουν από δεδομένα.
Θεμελιώδης στόχος της μηχανικής μάθησης είναι η *γενίκευση* ή η επαγωγική σύλληψη ενός άγνωστου κανόνα μέσα από παραδείγματα εφαρμογής του κανόνα.
Χαρακτηριστικό παράδειγμα μηχανικής εκμάθησης είναι το φιλτράρισμα ανεπιθύμητων μηνυμάτων.
Παρατηρώντας χιλιάδες μηνύματα ηλεκτρονικού ταχυδρομείου που είχαν προηγουμένως επισημανθεί ως ανεπιθύμητα ή επιθυμητά (αγγλκ. spam or ham), τα φίλτρα ανεπιθύμητης αλληλογραφίας μαθαίνουν να ταξινομούν νέα μηνύματα.

```{code-cell} ipython3
# Python >= 3.5
import sys
assert sys.version_info >= (3,5)

# Scikit-Learn >= 0.20
import sklearn
assert sklearn.__version__ >= "0.20"

import os

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import numpy as np
import pandas as pd
import scipy
```

Θα προμηθευτούμε τα δεδομένα για τα σπίτια στην California από τη [αυτή](https://github.com/ageron/handson-ml2/tree/master/datasets/housing) τη διεύθυνση.
Το εν λόγω αρχείο λέγεται "housing.csv" και έτσι θα το διατηρήσουμε και στο δικό μας project.

Επειδή είναι πιθανό τα δεδομένα στο συγκεκριμένο αρχείο να αλλάξουν κάποια στιγμή (δεν πρέπει να ξεχνάμε ότι χρησιμοποιούνται για τις ανάγκες ενός βιβλίου και μπορεί να υπάρχουν λάθη ή να χρειάζονται μεγαλύτερη επεξεργασία κ.λπ.), μπορούμε να δημιουργήσουμε μία συνάρτηση που να αναλαμβάνει να κάνει αυτόματα τις εργασίες δημιουργίας του αρχείου.

```{code-cell} ipython3
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```

Τώρα μπορούμε να "τραβήξουμε" κατευθείαν το αρχείο `housing.csv` από τη συγκεκριμένη διεύθυνση και να το αποθηκεύσουμε στη θέση `./datasets/housing`, απλά με την κλήση της συνάρτησης `fetch_housing_data`

```{code-cell} ipython3
fetch_housing_data()
```

Η παραπάνω συνάρτηση δημιουργεί το φάκελο `datasets/housing` στον τρέχοντα κατάλογο εργασίας (workspace), "κατεβάζει" το αρχείο `housing.tgz` και εξάγει από αυτό, το αρχείο `housing.csv`, μέσα στο συγκεκριμένο φάκελο.

+++

## Εισαγωγή δεδομένων

Το επόμενο βήμα είναι να εισάγουμε τα δεδομένα του `housing.csv` σε ένα Pandas dataframe με τη συνάρτηση [read_csv()](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) (το Pandas έχει εγγενή υποστήριξη για αρχεία `.csv`):

```{code-cell} ipython3
housing = pd.read_csv('datasets/housing/housing.csv')
```

Το dataframe `housing` που δημιουργήσαμε, περιλαμβάνει πλέον όλες τις εγγραφές του `.csv`. Μπορούμε να δούμε τις πρώτες 5 από αυτές με τη μέθοδο [head()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html) (Pandas):

```{code-cell} ipython3
housing.head()
```

## Διερεύνηση των δεδομένων

Μπορούμε πλέον να αρχίσουμε να μελετάμε τα δεδομένα μας ζητώντας από την Python και συγκεκριμένα από τις διάφορες βιβλιοθήκες, να μας παρουσιάσουν πληροφορίες γι αυτά.

+++

### Ιδιότητες (Attributes)

Με τη μέθοδο [info()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html?highlight=info#pandas.DataFrame.info) (Pandas) μπορούμε να δούμε τις επικεφαλίδες των στηλών του df.
Οι επικεφαλίδες (αν υπάρχουν) συνιστούν η κάθε μία, μία ιδιότητα ή χαρακτηριστικό των εγγραφών.

```{code-cell} ipython3
housing.info()
```

Κάθε γραμμή είναι μία εγγραφή και αναπαριστά μία περιοχή.
Συνολικά υπάρχουν {math}`20640` εγγραφές (αντίστοιχες περιοχές), κάθε μία από τις οποίες χαρακτηρίζεται από 10 ιδιότητες (features, attributes).
Αυτές φαίνονται στα παραπάνω δεδομένα και αφορούν στο γεωγραφικό μήκος (longitude), γ. πλάτος (latitude) κ.λπ.
Ο αριθμός των εγγραφών θεωρείται σχετικά μικρός για τα δεδομένα της ML αλλά είναι αρκετός για να ξεκινήσουμε.
Μία παρατήρηση που πρέπει να κάνουμε είναι ότι το feature `total_beds` έχει μόνο {math}`20433` μη μηδενικές τιμές που σημαίνει ότι από 207 περιοχές λείπει το συγκεκριμένο χαρακτηριστικό.
Αυτό είναι κάτι που πρέπει να διευθετηθεί στα υπό εξέταση δεδομένα (θα το κάνουμε στη συνέχεια).

Αν παρατηρήσουμε την περιγραφή των χαρακτηριστικών του πίνακα παραπάνω θα δούμε ότι όλα είναι αριθμητικού τύπου (`float64`) εκτός από το `ocean_proximity` που είναι τύπου `object` δηλαδή οποιαδήποτε μορφή αντικειμένου που αναγνωρίζει η Python.
Στην πραγματικότητα είναι τύπου text και συγκεκριμένα περιγράφει αν η περιοχή είναι κοντά ή όχι στον ωκεανό.
Αυτό είναι ένα *κατηγορικό χαρακτηριστικό* (categorical attribute) και αυτό μπορούμε να το εξακριβώσουμε εύκολα με τη μέθοδο [value_counts](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html?highlight=info#pandas.DataFrame.info) η οποία επιστρέφει ποια είδη κατηγοριών υπάρχουν και πως κατανέμονται οι εγγραφές σε αυτές τις κατηγορίες (πόσες περιοχές ανά κατηγορία):

```{code-cell} ipython3
housing["ocean_proximity"].value_counts()
```

### Περιγραφικά στατιστικά (descriptive statistics)

Η μέθοδος [describe()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html?highlight=describe#pandas.DataFrame.describe) (Pandas) επιστρέφει τα περιγραφικά στατιστικά στοιχεία (descriptive statistics) για κάθε ένα από τα αριθμητικού τύπου χαρακτηριστικά του dataframe:

```{code-cell} ipython3
housing.describe()
```

Όπως βλέπουμε στο συγκεντρωτικό πίνακα, όλα τα χαρακτηριστικά του dataframe (οι τίτλοι στηλών) απαριθμούνται (count) $20640$ φορές, όσες είναι δηλαδή και οι εγγραφές.
Αυτό δεν ισχύει για το χαρακτηριστικό `total_bedrooms` που απαριθμείται $20433$ φορές.
Αυτό σημαίνει ότι για 207 εγγραφές δεν υπήρχαν μετρήσεις και καταχωρήθηκαν σαν `null` (μηδενική τιμή, μη-ύπαρξη τιμής, μη-τιμή κ.λπ.).
Η Pandas αντιλαμβάνεται τις `null` τιμές αυτόματα και τις αγνοεί στους υπολογισμούς.

Από τα υπόλοιπα στατιστικά χαρακτηριστικά τα *min* και *max* είναι οι ελάχιστες και μέγιστες παρατηρούμενες τιμές κάθε χαρακτηριστικού ενώ *mean* είναι η μέση τιμή του.
Το *std* είναι η *τυπική απόκλιση*[^std] (standard deviation ή {math}`\sigma`) που έχουν οι παρατηρούμενες τιμές του κάθε χαρακτηριστικού, από τη mean τιμή.

[^std]: Η τυπική απόκλιση {math}`\sigma` είναι η τετραγωνική ρίζα της *διακύμανσης* (variance -- {math}`\sigma^2`), δηλαδή {math}`\sqrt{\sigma^2} = \sigma`.
Η διακύμανση είναι η μέση τετραγωνική απόκλιση από το μέσο όρο (mean).
Όταν παρατηρούμε ένα φαινόμενο που ακολουθεί την *κανονική κατανομή* (normal distribution) ή αλλιώς την *κατανομή Gauss* (gaussian distribution), που απεικονίζεται σαν μία "καμπάνα", ισχύει ο κανόνας του "68/95/99".
Ο κανόνας αυτός λέει ότι το {math}`68\%` περίπου των παρατηρήσεων του φαινομένου βρίσκεται σε αποστάσεις {math}`\pm 1 \sigma` εκατέρωθεν του mean, το {math}`95\%` σε αποστάσεις {math}`\pm 2 \sigma` και το {math}`97\%` σε αποστάσεις {math}`\pm 3 \sigma`.

Οι σειρές $25\%$, $50\%$ και $75\%$ δείχνουν τα αντίστοιχα *εκατοστημόρια* (percentiles).
Ένα εκατοστημόριο υποδεικνύει μία τιμή η οποία δηλώνει ότι, ένα δεδομένο ποσοστό παρατηρήσεων παρουσιάζει σε ένα χαρακτηριστικό, τιμή κάτω από μία συγκεκριμένη (οπότε εννοείται ότι το υπόλοιπο ποσοστό μέχρι το $100\%$ παρουσιάζει τιμές μεγαλύτερες από αυτή).
Πιο απλά, το 25% των περιοχών της California, παρουσιάζει `housing_median_age` (μέση ηλικία σπιτιών) μικρότερη από 18 έτη το $50\%$ είναι χαμηλότερο από 29 έτη και το $75\%$ είναι χαμηλότερο από 37.
Τα συγκεκριμένα εκατοστημόρια (25/50/75) ονομάζονται επίσης και *τεταρτημόρια* (quartiles, 1ο, 2ο, 3ο και 4ο τεταρτημόρια).
Ειδικά το 2ο τεταρτημόριο ($50\%$) ονομάζεται και *διάμεσος* (median).


```{code-cell} ipython3
%matplotlib inline
housing.hist(bins=50, figsize=(20,15))
plt.show;
```