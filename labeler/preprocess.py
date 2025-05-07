from typing import Dict, List, Optional, Union
from collections import OrderedDict

import re

import pandas as pd

from common import to_list
from pattern_matcher import ConjunctPatterns
from labeler_utils import dict_ordered_by_len, remove_after_substring
from string_utils import border_alphanumeric, replace

REPLACE = {
    "unbordered": { # Order of replacement is determined by string length - used to replace patterns even in the middle of words - need to be very careful
        # fix hyphen issues
        "-nodal ": " nodal ",
        "-degree ": " degree ",

        # fascicular, bifascicular, trifascicular, etc.
        "fasicular": "fascicular",
        "fasicuar": "fascicular",

        # artifact, artifacts, artifactual, etc.
        "artefact": "artifact",
        "artefacts": "artifacts",
        "artifactual": "artifact",
        "atifact": "artifact",

        # atrial / biatrial
        "atrialy": "atrial",
        "atrially": "atrial",
        "atrrial": "atrial",
        "atrialo": "atrial",
        "atiral": "atrial",
        "atril": "atrial",
        "atrail": "atrial",
        "attrial": "atrial",
        "arial": "atrial",
        "atial": "atrial",
        "atyrial": "atrial",

        # junctional
        "junctiional": "junctional",
        "junctrional": "junctional",
        "jucntional": "junctional",
        "junctioinal": "junctional",
        "juctional": "junctional",

        # ventricular / intraventricular / biventricular
        "ventricularly": "ventricular",
        "ventricularlly": "ventricular",
        "ventricularl": "ventricular",
        "ventricuar": "ventricular",
        "ventrcular": "ventricular",
        "ventriular": "ventricular",
        "ventricualr": "ventricular",
        "ventruicular": "ventricular",
        "venricular": "ventricular",
        "venrticular": "ventricular",
        "ventriclar": "ventricular",
        "ventircular": "ventricular",
        "ventircular": "ventricular",
        "ventriicular": "ventricular",
        "ventruclar": "ventricular",
        "venticular": "ventricular",
        "ventricuklar": "ventricular",
        "ventriucular": "ventricular",
        "ventrciualr": "ventricular",

        # lateral, inferoposterolateral, etc.
        "laterla": "lateral",
        "laterl": "lateral",
        "laterally": "lateral",
        "lateal": "lateral",
        "lareral": "lateral",

        # antero, anteroseptal, etc.
        "anetro": "antero",

        # septal, anteroseptal, etc.
        "septai": "septal",
        "seotal": "septal",
        "septaal": "septal",

        # aberrant, aberrantly, etc. - fix "bb", # of Rs, and "e" -> "a"
        "abber": "aber", # "bb" to "b"
        "aberen": "aberran", # 1 Rs, "e"
        "aberan": "aberran", # 1 Rs, "a"
        "aberren": "aberran", # "e" to "a"
        "aberrren": "aberran", # 3 Rs, "e"
        "aberrran": "aberran", # 3 Rs, "a"

        "`": " ",
    },
    "bordered": OrderedDict([
        ("electrocardiogram", "ecg"),
        ("ekg", "ecg"),

        ("recording", "tracing"),

        ("mismastch", "mismatch"),

        ("rule-out", "rule out"),
        ("r/oacute", "r/o acute"),
        ("r/ope", "r/o pe"),
        ("r/o", "rule out"),

        # rhythm, arrhythmia
        ("rhtyhm", "rhythm"),
        ("rhytm", "rhythm"),
        ("rhyhthm", "rhythm"),
        ("rhyuthm", "rhythm"),
        ("rhythmm", "rhythm"),
        ("shythm", "rhythm"),
        ("thythm", "rhythm"),
        ("rhythmi", "rhythm"),
        ("rhthm", "rhythm"),
        ("rjythm", "rhythm"),
        ("thrthm", "rhythm"),
        ("rhhthm", "rhythm"),

        # same as rhythm typos, but with "ar" prepended and "ia" appended
        ("arrhtyhmia", "arrhythmia"), 
        ("arrhytmia", "arrhythmia"),
        ("arrhyhthmia", "arrhythmia"),
        ("arrhyuthmia", "arrhythmia"),
        ("arrhythmmia", "arrhythmia"),
        ("arshythmia", "arrhythmia"),
        ("arthythmia", "arrhythmia"),
        ("arrhythmia", "arrhythmia"),
        ("arrhthmia", "arrhythmia"),
        ("arrjythmia", "arrhythmia"),
        ("arrhhthm", "arrhythmia"),

        # 1 "r"
        ("arhythmia", "arrhythmia"),
        ("arthrthmia", "arrhythmia"),


        # "beat" and "contraction" are not perfectly interchangeable, but contraction only appears in specifically these patterns: "premature ventricular contraction", "premature atrial contraction", "atrial premature contraction"
        # Here, they are interchangeable, so let's change them
        ("contraction", "beat"),
        ("contractions", "beats"),

        ("depol", "depolarization"),
        ("depolarisation", "depolarization"),

        # Shortforms
        ("lds", "leads"),
        ("depr", "depression"),
        ("avb", "atrioventricular block"),
        ("avblock", "atrioventricular block"),
        ("a-vblock", "atrioventricular block"),
        ("av-block", "atrioventricular block"),

        # ecg
        ("elctrode", "electrode"),

        # electronic pacemaker
        ("pacer", "pacemaker"),
        ("pacermaker", "pacemaker"),
        ("paceaker", "pacemaker"),
        ("pacemake", "pacemaker"),
        ("pacemker", "pacemaker"),
        ("paed", "paced"),

        ("elecrtonic", "electronic"),
        ("electonic", "electronic"),

        ("pseudo fusion", "pseudofusion"),
        ("pseudo-fusion", "pseudofusion"),

        ("captured", "capture"),

        ("non capture", "non-capture"),
        ("noncapture", "non-capture"),

        ("under sensing", "undersensing"),
        ("undersending", "undersensing"),

        ("undersensing pacemaker", "undersensing"),
        ("pacemaker undersensing", "undersensing"),
        ("pacemaker functional undersensing", "undersensing"),

        ("triggerred", "triggered"),
        ("triggere", "triggered"),
        ("triggered by", "due to"), # Remove so it doesn't turn into a "triggered" pacing false positive
        ("triggered from", "due to"), # Remove so it doesn't turn into a "triggered" pacing false positive

        ("sense", "sensing"),
        ("sensed", "sensing"),
        ("tracking", "sensing"),
        ("tracked", "sensing"),
        ("track", "sensing"),

        ("pacemaker capture", "capture"),
        ("pacemaker sensing", "sensing"),

        ("sensinhg", "sensing"),

        ("apcing", "pacing"),
        ("paocing", "pacing"),

        # dual-chamber pacing
        ("dual-paced", "dual-chamber pacing"),
        ("dual paced", "dual-chamber pacing"),
        ("dual pacing", "dual-chamber pacing"),
        ("dual pacemaker", "dual-chamber pacing"),
        ("dual chamber", "dual-chamber"),

        # location
        ("inf", "inferior"),
        ("inferioir", "inferior"),
        ("inferior", "inferior"),
        ("inferio", "inferior"),

        ("ant", "anterior"),
        ("anterioir", "anterior"),

        ("av", "atrioventricular"),
        ("a-v", "atrioventricular"),

        ("avn", "atrioventricular node"),

        ("iv", "intraventricular"),
        ("i.v.", "intraventricular"),
        ("i.v", "intraventricular"),

        ("supraven", "supraventricular"),
        ("superventricular", "supraventricular"),

        ("subendo", "subendocardial"),
        ("sub endo", "subendocardial"),
        ("sub-endo", "subendocardial"),
        ("sub-endocardial", "subendocardial"),

        ("rv", "right ventricular"),
        ("lv", "left ventricular"),

        ("extremity", "limb"),

        ("apica", "apical"),
        
        # biventricular
        ("biv", "biventricular"),
        ("bi-v", "biventricular"),
        ("bi v", "biventricular"),
        ("bi-vent", "biventricular"),
        ("bi- ventricular", "biventricular"),
        ("bi-ventricular", "biventricular"),
        ("bi ventricular", "biventricular"),
        ("biventricularly", "biventricular"),
        ("biventricuar", "biventricular"),
        ("biventrcular", "biventricular"),
        ("biventriular", "biventricular"),
        ("biventricualr", "biventricular"),
        ("biventruicular", "biventricular"),
        ("bivenricular", "biventricular"),
        ("bivent", "biventricular"),

        # "a-BiV" pacing
        ("a-biventricular paced", "atrial pacing, biventricular pacing"),
        ("a-biventricular pacing", "atrial pacing, biventricular pacing"),

        # ventricular / idioventricular intraventricular / biventricular
        ("entricular", "ventricular"),
        ("vent", "ventricular"),
        ("idiovent", "idioventricular"),
        ("ideoventricular", "idioventricular"),
        ("ideo-ventricular", "idioventricular"),
        ("ideo -ventricular", "idioventricular"),
        ("ideo- ventricular", "idioventricular"),
        ("ideo ventricular", "idioventricular"),
        ("idio-ventricular", "idioventricular"),
        ("idio -ventricular", "idioventricular"),
        ("idio- ventricular", "idioventricular"),
        ("idio ventricular", "idioventricular"),

        # atrial
        ("aatrial", "atrial"),
        ("artial", "atrial"),
        ("intratrial", "intraatrial"),

        ("la", "left atrial"),
        ("ra", "right atrial"),

        # lateral
        ("ateral", "lateral"),
        ("latera", "lateral"),

        # ventriculoatrial
        ("va", "ventriculoatrial"),
        ("v-a", "ventriculoatrial"),

        # reentry
        ("re-entry", "reentry"),
        ("re entry", "reentry"),
        ("reentrant", "reentry"),
        ("re-entrant", "reentry"),
        ("re entrant", "reentry"),

        ("re-entraant", "reentry"),
        ("reenetrant", "reentry"),
        ("re-enetrant", "reentry"),

        # focal
        ("uni-focal", "unifocal"),
        ("bi-focal", "bifocal"),
        ("multi-focal", "multifocal"),

        # junctional
        ("avj", "junctional"),

        # sinus/sinoatrial - more or less interchangable in this context, e.g., sinus node and SA node
        ("sa", "sinus"),
        ("sinoatrial", "sinus"),
        ("sino-atrial", "sinus"),
        ("san", "sinus node"),

        ("snus", "sinus"),
        ("sinuss", "sinus"),

        # tachyardia
        ("tach", "tachycardia"),
        ("tachy", "tachycardia"),
        ("tachyarrhythmia", "tachycardia"),
        ("tachycardy", "tachycardia"),

        ("tachcyardia", "tachycardia"),
        ("tachycrdia", "tachycardia"),
        ("taachycardia", "tachycardia"),
        ("tachycardai", "tachycardia"),
        ("tahycardia", "tachycardia"),

        ("stach", "sinus tachycardia"),

        ("wide-complex", "wide complex"),
        ("narrow-complex", "narrow complex"),

        # Have to carefully replace all the tachycardia acronyms/synonyms to responsibly parse e.g., monomorphic/polymorphic, sustained/nonsustained, paroxysmal/nonparoxysmal, persistant, permanent
        ("vt", "ventricular tachycardia"),
        ("pvt", "polymorphic ventricular tachycardia"),

        ("svt", "supraventricular tachycardia"),
        ("svta", "supraventricular tachycardia"),
        ("nsvt", "nonsustained supraventricular tachycardia"),
        ("psvt", "paroxysmal supraventricular tachycardia"),

        ("atrioventricular nodal reentry tachycardia", "avnrt"),
        ("atrioventricular node reentry tachycardia", "avnrt"),
        ("atrioventricular nodal reentry", "avnrt"),
        ("atrioventricular node reentry", "avnrt"),
        ("avnrt", "atrioventricular nodal reentry tachycardia"),

        ("atrioventricular reentry tachycardia", "avrt"),
        ("atrioventricular reentry", "avrt"),
        ("avrt", "atrioventricular reentry tachycardia"),

        ("bbr", "bundle branch reentry"),
        ("bundle branch reentry ventricular tachycardia", "bbrvt"),
        ("bundle branch reentry tachycardia", "bbrvt"),
        ("bundle branch reentry", "bbrvt"),
        ("bbrvt", "bundle branch reentry ventricular tachycardia"),

        ("bi-directional", "bidirectional"),
        ("bvt", "bidirectional ventricular tachycardia"),
        ("bivt", "bidirectional ventricular tachycardia"),

        ("torsade de pointes", "tdp"),
        ("torsades de pointes", "tdp"),
        ("torsades", "tdp"),
        ("torsade ventricular tachycardia", "tdp"),
        ("tdp", "torsade de pointes"),

        ("atach", "atrial tachycardia"),
        ("mat", "multifocal atrial tachycardia"),
        ("pat", "paroxysmal atrial tachycardia"),

        ("out flow", "outflow"),

        ("rvot", "right ventricular outflow tract"),
        ("out flow", "outflow"),

        ("poly", "polymorphic"),

        # fibrillation
        ("fib", "fibrillation"),
        ("flbrillation", "fibrillation"),
        ("fibrilaltion", "fibrillation"),
        ("fibrllation", "fibrillation"),

        ("af", "atrial fibrillation"),
        ("afib", "atrial fibrillation"),
        ("a-fib", "atrial fibrillation"),
        ("afibrillation", "atrial fibrillation"),
        ("a fibrillation", "atrial fibrillation"),
        ("atria fibrillation", "atrial fibrillation"),

        ("vf", "ventricular fibrillation"),
        ("vfib", "ventricular fibrillation"),
        ("v-fib", "ventricular fibrillation"),
        ("vfibrillation", "ventricular fibrillation"),
        ("v fibrillation", "ventricular fibrillation"),

        # flutter
        ("fluter", "flutter"),
        ("flut", "flutter"),

        ("afl", "atrial flutter"),
        ("aflutter", "atrial flutter"),
        ("a-flutter", "atrial flutter"),
        ("a flutter", "atrial flutter"),
        ("atria flutter", "atrial flutter"),

        ("vflutter", "ventricular flutter"),
        ("v-flutter", "ventricular flutter"),
        ("v flutter", "ventricular flutter"),


        # bradycardia
        ("brady", "bradycardia"),
        ("bradyarrhythmia", "bradycardia"),
        ("bradycardy", "bradycardia"),

        # heart disease
        ("heartdisease", "heart disease"),
        ("cong.", "congenital"),
        ("electriclal", "electrical"),

        # abnormal / abnormality
        ("abn", "abnormality"),

        ("unusual", "abnormal"),
        ("abnormnal", "abnormal"),
        ("abnrm", "abnormality"),
        ("abn", "abnormality"),
        ("abnormaliy", "abnormality"),
        ("abnornmality", "abnormality"),
        ("abnormalitry", "abnormality"),
        ("abnorm,ality", "abnormality"),
        ("abnorma,ity", "abnormality"),
        ("abhnormality", "abnormality"),
        ("abnomality", "abnormality"),
        ("abormality", "abnormality"),
        ("abnoramlity", "abnormality"),
        ("anormality", "abnormality"),

        ("repol", "repolarization"),
        ("repolarisation", "repolarization"),
        ("repolrization", "repolarization"),
        ("replorization", "repolarization"),
        ("repolarizatioon", "repolarization"),
        ("repoloraization", "repolarization"),
        ("repolaristaion", "repolarization"),
        ("repolchanges", "repolarization changes"),

        ("ear;y", "early"),

        ("st and t", "st-t"),
        (" st t ", " st-t "),
        ("st and st", "st-t"),
        ("st and marked t", "marked st-t"),
        ("st and marked st", "marked st-t"),

        ("st segment elevation", "st elevation"),
        ("st-elevation", "st elevation"),

        # heart/heart rate
        ("hr", "heart rate"),
        ("heartrate", "heart rate"),

        ("heat", "heart"),

        ("patten", "pattern"),
        ("psttern", "pattern"),
        ("patteren", "pattern"),
        ("patterrn", "pattern"),
        ("patetrn", "pattern"),
        ("-pattern", " pattern"),

        # ischemia
        ("ischaemia", "ischemia"), # Alternative spelling, not a typo
        ("ishemia", "ischemia"),
        ("ischema", "ischemia"),
        ("ischemai", "ischemia"),
        ("ischemis", "ischemia"),
        ("ishcemia", "ischemia"),
        ("ichemia", "ischemia"),
        ("icshemia", "ischemia"),
        ("iscemia", "ischemia"),
        ("oschemia", "ischemia"),

        ("hyperlalemia", "hyperkalemia"),

        # infarction
        ("infarction", "infarct"), # standardize
        ("mi", "infarct"),
        ("mis", "infarct"),
        ("myocardial infarction", "infarct"),

        ("ami", "acute infarct"),

        # axis deviation
        ("north-west", "northwest"),
        ("north -west", "northwest"),
        ("north- west", "northwest"),
        ("north west", "northwest"),

        ("lefward", "leftward"),
        ("leeft", "left"),
        ("devaition", "deviation"),

        # conduction
        ("coduction", "conduction"),
        ("conducion", "conduction"),
        ("conudction", "conduction"),
        ("conducton", "conduction"),
        ("conuction", "conduction"),
        ("cond", "conduction"),
        ("condicted", "conducted"),
        ("conduceted", "conducted"),
        ("ashman's", "ashman"),

        ("rettrograde", "retrograde"),

        ("vcd", "ventricular conduction delay"),

        # conduction block
        ("bolck", "block"),
        ("blcok", "block"),
        ("blocl", "block"),
        ("bl;ock", "block"),
        ("b lock", "block"),
        ("bloick", "block"),
        ("bock", "block"),
        ("blocck", "block"),

        ## conduction ratios
        ("2 to 1", "2:1"), # Checked for others like this using `diagnoses.str.extract(r'(\d\s*to\s*\d)')`, but didn't find any
        ("2-to-1", "2:1"), # TODO - check
        ("4-to-1", "4:1"),

        ("2: 1", "2:1"),


        ("heart block", "atrioventricular block"), # Another term for AV block
        ("heart-block", "atrioventricular block"),
        ("chb", "complete atrioventricular block"),
        ("complete hb", "complete atrioventricular block"),
        ("comp-hb", "complete atrioventricular block"),
        ("hb", "atrioventricular block"),
        ("complete-atrioventricular block", "complete atrioventricular block"),

        ("atrioventricular-block", "atrioventricular block"),
        ("atrioventricular-node", "atrioventricular nodal"),
        ("atrioventricular-nodal", "atrioventricular nodal"),
        ("atrioventricular node block", "atrioventricular nodal block"),
        ("nodal atrioventricular block", "atrioventricular nodal block"),
        ("atrioventricular nodal block", "atrioventricular block"), # TODO - Remove and parse properly if clinically relevant
        ("nodal block", "block"), # Let's actually remove all references to nodal blocks for now

        ("hemi-block", "hemiblock"),
        ("hemi -block", "hemiblock"),
        ("hemi- block", "hemiblock"),
        ("hemi block", "hemiblock"),
        ("bi-fascicular", "bifascicular"),
        ("tri-fascicular", "trifascicular"),

        # standardize conduction references (delays, blocks)
        ("conduction delay", "delay"),
        ("conduction block", "block"),

        # type roman numerals
        ("type i", "type 1"),
        ("type ii", "type 2"),
        ("type iii", "type 3"),

        # mobitz
        ("mobitx", "mobitz"),

        ("mobitz i", "mobitz type 1"),
        ("mobitz ii", "mobitz type 2"),
        ("mobitz 1", "mobitz type 1"),
        ("mobitz 2", "mobitz type 2"),
        ("type 1 mobitz", "mobitz type 1"),
        ("type 2 mobitz", "mobitz type 2"),

        ("atrioventricular nodal mobitz type 1", "atrioventricular block mobitz type 1"),

        # wenckebach
        ("wenchebach", "wenckebach"),
        ("wenchbach", "wenckebach"),
        ("wenckbach", "wenckebach"),
        ("wenkebach", "wenckebach"),
        ("wencheback", "wenckebach"),
        ("wenckeback", "wenckebach"),

        ("wenckebach", "mobitz type 1"),
        ("mobitz type 1 (mobitz type 1)", "mobitz type 1"), # Fixing e.g., "mobitz type 1 (wenckebach)" / "wenckebach (mobitz type 1)", which would now be "mobitz type 1 (mobitz type 1)"

        # ectopics
        ("ectopuc", "ectopic"),
        ("ectopc", "ectopic"),
        ("ectoic", "ectopic"),

        ("extrasystole", "ectopic beat"),
        ("extrasystoles", "ectopic beats"),

        ("premature ectopic", "premature"),
        ("preamture", "premature"),
        ("pemature", "premature"),
        ("permature", "premature"),

        ("eacape", "escape"),

        ("bigemini", "bigeminy"),
        ("bigmeny", "bigeminy"),
        ("begeminy", "bigeminy"),
        ("bigimini", "bigeminy"),
        ("bigiminy", "bigeminy"),
        ("bigemibal", "bigeminal"),

        ("trigemini", "trigeminy"),
        ("trigmeny", "trigeminy"),
        ("trigimini", "trigeminy"),
        ("trigiminy", "trigeminy"),

        ("escape-capture", "escape capture"),

        ("accel", "accelerated"),
        ("acceleated", "accelerated"),

        # aberrant, aberrantly, etc.
        # typos
        ("aberrabcy", "aberrancy"),
        ("aberrany", "aberrancy"),

#         # "e" to "a"
#         ("aberrent", "aberrant"),
#         ("aberrency", "aberrancy"),
#         ("aberrently", "aberrantly"),

#         # "bb" to "b"
#         ("abberrant", "aberrant"),
#         ("abberrent", "aberrant"),
#         ("abberrancy", "aberrancy"),
#         ("abberrency", "aberrancy"),
#         ("abberrantly", "aberrantly"),
#         ("abberrently", "aberrantly"),

        # pulmonary embolism
        ("pe", "pulmonary embolism"),
        ("pulm", "pulmonary"),


        # aberrantly conducted
        ("aberrantly-conducted", "aberrated"),
        ("aberrantly conducted", "aberrated"),
        ("aberrant-conducted", "aberrated"),
        ("aberrant conducted", "aberrated"),
        ("aberrantly ventricular conducted", "aberrated"),

        # aberrantly conducting
        ("aberrantly-conducting", "aberrated"),
        ("aberrantly conducting", "aberrated"),
        ("aberrant-conducting", "aberrated"),
        ("aberrant conducting", "aberrated"),
        ("aberrantly ventricular conducting", "aberrated"),

        # standardize patterns into "aberrancy"
        ("aberrant ventricular conduction", "aberrancy"),
        ("aberrant conduction", "aberrancy"),
        ("aberration", "aberrancy"),


        ("cocealed", "concealed"),
        ("accessorypathway", "accessory pathway"),

        # standardize patterns into "undetermined"
        ("indeterminate", "undetermined"),
        ("indeterm", "undetermined"),

        ("unknow", "unknown"),
        ("unknownj", "unknown"),
        ("unknown", "undetermined"),

        # brugada
        ("brugada s", "brugada"),
        ("brugada's", "brugada"),
        ("brugadas", "brugada"),
        ("bridada", "brugada"),
        ("btrugada", "brugada"),
        ("bruagada", "brugada"),
        ("brugad", "brugada"),
        ("brigada", "brugada"),

        ("wall damage", "infarct"),


        # triple compound regions not fixed by conjunct patterns
        ("inferopostero-lateral", "inferoposterolateral"),

        # wave
        ("wavew", "wave"),
        ("wavea", "waves"),
        ("wav e", "wave"),
        ("wav", "wave"),
        ("wve", "wave"),
        ("wves", "waves"),
        ("wavre", "wave"),

        ("natiev", "native"),
        ("atypcal", "atypical"),
        ("attypical", "atypical"),

        ("elev", "elevation"),
        ("elevantion", "elevation"),

        ("non-conducted", "nonconducted"),
        ("non- conducted", "nonconducted"),
        ("non -conducted", "nonconducted"),
        ("non - conducted", "nonconducted"),
        ("non conducted", "nonconducted"),
        ("blocked", "nonconducted"),
        ("dropped", "nonconducted"),

        # beats, complexes, salvos, runs, etc.
        ("compelx", "complex"),
        ("comple", "complex"),
        ("compleses", "complexes"),
        ("complxes", "complexes"),
        ("coplexes", "complexes"),
        ("compelxes", "complexes"),

        ("beays", "beats"),

        ("broad", "wide"), # Always referring to a wide QRS complex, though sometimes in patterns like "broad RBBB"
        ("qide", "wide"),

        ("narroe", "narrow"),

        ("wide-qrs", "wide qrs"),
        ("narrow-qrs", "narrow qrs"),

        ("salvoes", "salvos"),

        ("groupe", "grouped"),

        ("artifct", "artifact"),
        ("failur", "failure"),
        ("interogation", "interrogation"),

        # Bundle branch block
        ("byndle", "bundle"),
        ("buble", "bundle"),

        ("bundle-branch", "bundle branch"),
        ("bbb", "bundle branch block"),

        ("rbbb", "right bundle branch block"),
        ("lbbb", "left bundle branch block"),

        ("rbbbb", "right bundle branch block"),
        ("lbbbb", "left bundle branch block"),

        ("rbb b", "right bundle branch block"),
        ("lbb b", "left bundle branch block"),

        # Note: Already replacing, e.g., "right bundle block" with "right bundle branch block" in post
        ("right branch block", "right bundle branch block"),
        ("left branch block", "left bundle branch block"),

        ("incomplete rbb", "incomplete right bundle branch block"),
        ("incomplete lbb", "incomplete left bundle branch block"),
        ("irbbb", "incomplete right bundle branch block"),
        ("irbb", "incomplete right bundle branch block"),
        ("ilbbb", "incomplete left bundle branch block"),
        ("ilbb", "incomplete left bundle branch block"),
        ("incrbb", "incomplete right bundle branch block"),
        ("incrbbb", "incomplete right bundle branch block"),
        ("incompleterbbb", "incomplete right bundle branch block"),
        ("inclbb", "incomplete left bundle branch block"),
        ("inclbbb", "incomplete left bundle branch block"),
        ("incompletelbbb", "incomplete left bundle branch block"),
        ("inc", "incomplete"),

        ("crbbb", "complete right bundle branch block"),
        ("crbb", "complete right bundle branch block"),
        ("clbbb", "complete left bundle branch block"),
        ("clbb", "complete left bundle branch block"),

        # TODO - double-check these - does anyone say "RBB"/"LBB" to reference anything other than a block?
        ("rbb", "right bundle branch block"),
        ("lbb", "left bundle branch block"),

        ("multipe", "multiple"),

        ("eleveated", "elevated"),
        ("prolomged", "prolonged"),
        ("pronlonged", "prolonged"),
        ("proloned", "prolonged"),
        ("ptolonged", "prolonged"),
        ("prolomged", "prolonged"),
        ("pronlonged", "prolonged"),
        ("prolonge", "prolonged"),
        ("prongation", "prolongation"),
        ("prolongation", "prolongation"),

        # ectopic beat acroynms
        ("ve", "ventricular ectopic"),
        ("veb", "ventricular ectopic beat"),
        ("vebs", "ventricular ectopic beats"),

        # premature beat acroynms
        ("pac", "premature atrial beat"),
        ("pacs", "premature atrial beats"),
        ("apb", "premature atrial beat"),
        ("apbs", "premature atrial beats"),
        ("apc", "premature atrial beat"),
        ("apcs", "premature atrial beats"),

        ("pvc", "premature ventricular beat"),
        ("pvcs", "premature ventricular beats"),
        ("vpb", "premature ventricular beat"),
        ("vpbs", "premature ventricular beats"),

        ("psc", "premature supraventricular beat"),
        ("pscs", "premature supraventricular beats"),
        ("svpb", "premature supraventricular beat"),
        ("svpbs", "premature supraventricular beats"),

        ("pjc", "premature junctional beat"),
        ("pjcs", "premature junctional beats"),


        # infraction
        ("infarction", "infarct"),
        ("infract", "infarct"),
        ("infraction", "infarct"),

        ("devaition", "deviation"),

        # leads
        ("precrdial", "precordial"),
        ("chest lead", "precordial lead"),
        ("chest leads", "precordial leads"),
        ("malposition", "misplacement"),
        ("malplacement", "misplacement"),
        ("misalignment", "misplacement"),
        ("lead location", "lead placement"),
        ("lead position", "lead placement"), 

        # descriptor synonyms
        ("fast", "rapid"),
        ("elevated", "rapid"),
        ("abbreviated", "short"),
        ("shortened", "short"),

        ("varying degrees of", "variable"),
        ("varying", "variable"),

        ("unable to confirm", "cannot confirm"),

        # descriptor typos
        ("intermittant", "intermittent"),
        ("intermmitent", "intermittent"),
        ("undetermine ", "undetermined "),
        ("uniterpretable", "uninterpretable"),
        ("intepretation", "interpretation"),
        ("progressionm", "progression"),
        ("promnent", "prominent"),
        ("prominant", "prominent"),
        ("promenent", "prominent"),
        ("ptominant", "prominent"),
        ("ocasional", "occasional"),
        ("occ", "occasional"),
        ("vaiable", "variable"),
        ("vairable", "variable"),
        ("variale", "variable"),
        ("varoable", "variable"),
        ("variabl;e", "variable"),
        ("pooir", "poor"),
        ("por", "poor"),
        ("poofr", "poor"),
        ("dalayed", "delayed"),
        ("signficant", "significant"),
        ("frquent", "frequent"),
        ("isoalted", "isolated"),
        ("hig", "high"),
        ("higgh", "high"),
        ("secon", "second"),
        ("sebcondary", "secondary"),
        ("completer", "complete"),
        ("difuse", "diffuse"),
        ("consistant", "consistent"),
        ("consisitent", "consistent"),
        ("inconsistant", "inconsistent"),
        ("prevoius", "previous"),

        ("channges", "changes"),

        # degrees (blocks)
        ("1*atrioventricular", "1st degree atrioventricular"),
        ("1*", "1st degree"),
        ("1 degree", "1st degree"),
        ("first degree", "1st degree"),
        ("!*", "1st degree"), # Typo - didn't press shift on the 1

        ("2*atrioventricular", "2nd degree atrioventricular"),
        ("2*", "2nd degree"),
        ("2 degree", "2nd degree"),
        ("second degree", "2nd degree"),

        ("3*atrioventricular", "3rd degree atrioventricular"),
        ("3*", "3rd degree"),
        ("3 degree", "3rd degree"),
        ("third degree", "3rd degree"),

        # Remove

        # redundant text removal
        (" present ", " "),
        ("but", ""),
        ("wall", ""),

        # interval
        ("qt u", "qtu"),
        ("qt-u", "qtu"),
        ("qt- u", "qtu"),
        ("qt -u", "qtu"),

        ("qtc", "qt"), # Standardize for the purposes of pattern matching - seem to be used interchangeably despite their definitions

        # remove of 'interval' from named intervals to help standardize
        ("qt interval", "qt"),
        ("qt-interval", "qt"),
        ("pr interval", "pr"),
        ("pr-interval", "pr"),
        ("rp interval", "rp"),
        ("rp-interval", "rp"),
        ("qtu interval", "qtu"),
        ("qtu-interval", "qtu"),

        ("long-rp", "short rp"),
        ("short-rp", "short rp"),

        # Hyphens/spaces
        ("non-specific", "nonspecific"),
        ("non specific", "nonspecific"),
        ("pre-excitation", "preexcitation"),
        ("pre excitation", "preexcitation"),
        ("peexcitation", "preexcitation"),
        ("unsustained", "nonsustained"),
        ("non sustained", "nonsustained"),
        ("non-sustained", "nonsustained"),
        ("non -sustained", "nonsustained"),
        ("non- sustained", "nonsustained"),
        ("ns", "nonsustained"),
        ("-noise", " noise"),
        ("s1-s2-s3", "s1s2s3"),
        ("rate related", "rate-related"),
        ("post conversion", "post-conversion"),
        ("clock-wise", "clockwise"),
        ("counter-clockwise", "counterclockwise"),
        ("pre-existing", "preexisting"),
        ("pre existing", "preexisting"),
        ("qrst", "qrs-t"),
        ("high-grade", "high grade"),
        ("high-degree", "high grade"),
        ("high degree", "high grade"),
        ("non-paroxysmal", "nonparoxysmal"),
        ("non paroxysmal", "nonparoxysmal"),
        ("non- paroxysmal", "nonparoxysmal"),
        ("non -paroxysmal", "nonparoxysmal"),

        ("j-point", "j point"),

        ("undetermined rhythm ?", "undetermined rhythm ?-"), # Identifies the coming rhythm as a guess
        ("+ ?", "+?"),
        (", may be normal variant", " may be normal variant"),

        # uncertainty descriptors
        ("prob", "probably"),
        ("probalbe", "probable"),
        ("probale", "probable"),
        ("probale", "probable"),
        ("probaly", "probably"),
        ("likley", "likely"),
        ("likely", "probably"), # Turn "likely" into the equivalent "probably"
        ("most probably", "probably"), # Turn "most probably"/"most likely" into just "probably"

        # connectives
        ("v.s", "versus"),
        ("v.s.", "versus"),
        ("vs", "versus"),
        ("vs.", "versus"),

        ("+/_", "+/-"),
        ("cannout", "cannot"),
        ("excldue", "exclude"),
        ("sugeests", "suggests"),
        ("suggestd", "suggests"),
        ("aspreviously", "as previously"),
        ("iwth", "with"),
        ("witgh", "with"),
        ("cannt", "cannot"),
        ("alternatres", "alternates"),
        ("appeas", "appears"),
        ("competeing", "competing"),
        ("consdier", "consider"),

        # Stop connectives from being parsed as descriptors
        ("more probably than", "morelikelythan"),
        ("or more probably", "ormorelikely"),

        # Has ` instead of whitespace
        ("`", " "),
    ]),
    # Order of appearence becomes the order of replacement
    "regex": OrderedDict([
        # Standardize " -? age", " -?age", " ? age", " ?age", "- ? age", "- ?age", "-- ? age", "--? age", "--?age", "-? age", "-?age", "?age"
        (r"-?\s?-?\s?\?\s?age", " ? age "),

        # "?" at the beginning of the string
        (r"^\?", "<before?> "),

        # "-?", "--?"
        (r"-?-\?", " <after?> "),

        # Add whitespace after comma - does not replace the non-whitespace character
        (r",(\S)", r", \1"),

        # Get rid of extra spaces
        (r"\s+", " "),
    ]),
}

# === ConjPattern replacements ===
ASSESS_PACEMAKER_GROUPS = [
    [ # 0
        "suggest",
        "advise",
        "need",
        "needs",
        "recommend",
    ],
    [ # 1
        "assessment",
        "check",
        "evaluation",
        "interrogation",
    ],
    [ # 2
        "assess",
        "assessing",
        "check",
        "checking",
        "evaluate",
        "evaluating",
    ],
]

PACER_TYPES = {
    # Atrial
    "atrial": "atrial",
    "a": "atrial",

    # Ventricular
    "ventricular": "ventricular",
    "v": "ventricular",

    # Biventricular
    "biv": "biventricular",
    "biventricular": "biventricular",

    # Atrioventricular
    "atrioventricular": "atrioventricular",

    # Dual-chamber
    "dual-chamber": "dual-chamber",

    # Atrioventricular sequential
    "atrioventricular dual-paced": "atrioventricular sequential",
    "atrioventricular sequential": "atrioventricular sequential",
    "sequential atrioventricular": "atrioventricular sequential",

    # Pacing modes
    "vvi": "vvi",
    "ddd": "ddd",
    "aai": "aai",
}

PACING_GROUPS = [
    PACER_TYPES, # 0
    { # 1
        "pacing": "pacing",
        "pace": "pacing",
        "paced": "pacing",
        "pacemaker": "pacing",
        "pacemaker detected": "pacing",
    },
    { # 2 - Remove these
        "rhythm": "",
        "beat": "",
        "beats": "",
        "complex": "",
        "complexes": "",
        "pacemaker": "",
    },
]
PACING_TEMPLATES = {
    "paced [0] [2]": "[0] pacing",
    "paced [0]-[2]": "[0] pacing",
    "paced [0]- [2]": "[0] pacing",
    "paced [0] -[2]": "[0] pacing",
    "[0] paced [2]": "[0] pacing",
    "[0]-paced [2]": "[0] pacing",
    "[0] [1]": "[0] pacing",
    "[0]-[1]": "[0] pacing",
    "[0]- [1]": "[0] pacing",
    "[0] -[1]": "[0] pacing",
    "[0] [1] rhythm": "[0] pacing",
    "[0]-[1] rhythm": "[0] pacing",
    "[0]- [1] rhythm": "[0] pacing",
    "[0] -[1] rhythm": "[0] pacing",
    "demand [0] [1]": "[0] pacing <demand pacing>",
    "demand [0]-[1]": "[0] pacing <demand pacing>",
    "demand [0] -[1]": "[0] pacing <demand pacing>",
    "demand [0]- [1]": "[0] pacing <demand pacing>",
}


PACER_ENT_GROUPS = [
    { # 0
        "electronic pacemaker": "electronic pacemaker",
        "pacemaker": "electronic pacemaker",
        "pacing": "electronic pacemaker",
        "paced": "electronic pacemaker",
    },
    { # 1
        "spike": "spike",
        "spikes": "spikes",
        "artifact": "artifact",
        "artifacts": "artifacts",
        "beat": "",
        "beats": "",
        "complex": "",
        "complexes": "",
        "malfunction": "malfunction",
        "malfunctions": "malfunctions",
    },
]

PACER_ENT_TEMPLATES = {"[0] [1]": "[0] [1]"}

WAVE_ABNORMALITY_GROUPS = [
    { # 0
        "t": "t",
        "st": "st",
        "st-t": "st-t",
        "r": "r",
    },
]
WAVE_ABNORMALITY_TEMPLATES = {
    "[0] abnormality": "[0] wave abnormality",
    "[0] abnormalities": "[0] wave abnormalities",
    "[0] wave abnormality": "[0] wave abnormality",
    "[0] wave abnormalities": "[0] wave abnormalities",
    "abnormal [0]": "[0] wave abnormality",
    "abnormal [0] wave": "[0] wave abnormality",
    "abnormal [0] waves": "[0] wave abnormalities",

    "[0] changes": "[0] wave changes",
    "[0] change": "[0] wave changes",
}


ATRIAL_SLASH_GROUPS = [
    { # 0
        "flutter": "flutter",
        "fibrillation": "fibrillation",
        "tachycardia": "tachycardia",
        "capture": "capture",
        "sensing": "sensing",
    }
]
ATRIAL_SLASH_TEMPLATES = {
    "atrial [0]/[0]": "atrial [0] atrial [1]",
    "atrial [0] or [0]": "atrial [0] atrial [1]",
    "atrial [0] [0]": "atrial [0] atrial [1]",
    "atrial [0]-[0]": "atrial [0] atrial [1]",
}

# "infarct - acute" -> "infarct, acute"
# "st depression -possible rate related" -> "st depression, possibly rate-related"
AFTER_DESCRIPTOR_GROUPS = [
    { # 0 - Connector
        ", ": ", ",
        "-": ", ",
        "- ": ", ",
        " -": ", ",
        " - ": ", ",
    },
    { # 1 - Uncertainty
        "probably": "probably",
        "probable": "probably",
        "most probably": "probably",
        "possibly": "possibly",
        "possible": "possibly",
    },
    { # 2 - Descriptor
        "normal": "normal",
        "abnormal": "abnormal",
        "borderline": "borderline",
        "acute": "acute",
        "biventricular": "biventricular",
        "accelerated": "accelerated",
        "recent": "recent",
        "old": "old",
        "rate-related": "rate-related",
    },
]

AFTER_DESCRIPTOR_TEMPLATES = {
    "[0][1] [2]": "[0][1] [2]",
    "[0][2]": "[0][1]",
}

PREMATURE_GROUPS = [
    { # 0
        "ventricular": "ventricular",
        "supraventricular": "supraventricular",
        "atrial": "atrial",
        "junctional": "junctional",
    },
    { # 1
        "contraction": "beat",
        "contractions": "beats",
        "complex": "beat",
        "complexes": "beats",
        "beat": "beat",
        "beats": "beats",
        "depolarization": "beat",
        "depolarizations": "beats",

        "rhythm": "rhythm",
        "arrhythmia": "rhythm",
    },
]

PREMATURE_TEMPLATES = {
    "premature [0] [1]": "premature [0] [1]", # "premature junctional complexes" -> "premature junctional beats"
    "[0] premature [1]": "premature [0] [1]", # "atrial premature beat" -> "premature atrial beat"
    "[0] prematures": "premature [0] beats", # "junctional prematures" -> "junctional premature beats"
    "premature [1]": "premature [0]", # "premature contractions" -> "premature beats"
}

ECTOPIC_GROUPS = [
    { # 0
        "ventricular": "ventricular",
        "idioventricular": "idioventricular",
        "supraventricular": "supraventricular",
        "atrial": "atrial",
        "junctional": "junctional",
    },
    { # 1
        "contraction": "beat",
        "contractions": "beats",
        "complex": "beat",
        "complexes": "beats",
        "beat": "beat",
        "beats": "beats",
        "depolarization": "beat",
        "depolarizations": "beats",

        "rhythm": "rhythm",
        "arrhythmia": "rhythm",
    },
]

ECTOPIC_TEMPLATES = {
    "ectopic [0] [1]": "ectopic [0] [1]", # "ectopic junctional complexes" -> "ectopic junctional beats"
    "[0] ectopic [1]": "ectopic [0] [1]", # "atrial ectopic beat" -> "ectopic atrial beat"
    "[0] ectopics": "ectopic [0] beats", # "junctional ectopics" -> "junctional ectopic beats"
    "[0] ectopy": "ectopic [0] beats", # "junctional ectopy" -> "junctional ectopic beats"
    "ectopic [1]": "ectopic [0]", # "ectopic contractions" -> "ectopic beats"
    "[0] [1]": "[0] [1]", # "supraventricular complexes" -> "supraventricular beats"
}


INTERVAL_LENGTH_GROUPS = [
    { # 0
        "prolonged": "prolonged",
        "long": "prolonged",
        "prolongation of": "prolonged",
        "prolongation of the": "prolonged",

        "short": "short",
        "shortening of": "short",
    },
    { # 1
        "prolonged": "prolonged",
        "prolongation": "prolonged",
        "has lengthened": "prolonged",
        "appears prolonged": "prolonged",
        "is prolonged": "prolonged",
        "appears long": "prolonged",
        "is long": "prolonged",

        "short": "short",
        "shortening": "short",
        "has shortened": "short",
        "appears short": "short",
        "is short": "short",

    },
    { # 2
        "qt": "qt",
        "qtc": "qt",
        "pr": "pr",
        "qtu": "qtu",
    },
]
INTERVAL_LENGTH_PATTERNS = {
    "[0] [2]": "[0] [1]", # "prolonged QT" -> "prolonged QT"
    "[2] [1]": "[1] [0]", # "QTU appears short" -> "short QTU"

}

CONJPATS = [
    ("pacing", [PACING_GROUPS, PACING_TEMPLATES]),
    ("pacer entities", [PACER_ENT_GROUPS, PACER_ENT_TEMPLATES]),
    ("wave abnormality", [WAVE_ABNORMALITY_GROUPS, WAVE_ABNORMALITY_TEMPLATES]),
    ("atrial slash", [ATRIAL_SLASH_GROUPS, ATRIAL_SLASH_TEMPLATES]),
    ("after descriptor", [AFTER_DESCRIPTOR_GROUPS, AFTER_DESCRIPTOR_TEMPLATES]),
    ("premature", [PREMATURE_GROUPS, PREMATURE_TEMPLATES]),
    ("ectopic", [ECTOPIC_GROUPS, ECTOPIC_TEMPLATES]),
    ("interval length", [INTERVAL_LENGTH_GROUPS, INTERVAL_LENGTH_PATTERNS]),
]
CONJPAT_RESULTS = {}
CONJPAT_REPLACE_RESULTS = {}


def pre_replacement(texts):
    # Get rid of "confirmed/reconfirmed by physician"/"compared to the previous ECG"
    remove_after = [
        "also confirmed by",
        "reconfirmed by",
        "confirmed by",

        # May get rid of relevant info, but avoids labels attributing to previous ECG
        "compared to the ecg performed", 
        "compared to the previous ecg",
        "when compared to the previous ecg",
        "no significant change when compared to the previous ecg",
    ]
    remove_after = sorted(remove_after, key=len, reverse=True)

    for substring in remove_after:
        texts = remove_after_substring(texts, substring)

    texts = texts.str.strip()

    # Case-sensitive replacements necessary to make before converting to lowercase
    # AT -> atrial tachycardia
    texts = texts.str.replace(
        border_alphanumeric("AT"),
        "atrial tachycardia",
        regex=True,
        case=True,
    )
    # EAT -> ectopic atrial tachycardia
    texts = texts.str.replace(
        border_alphanumeric("EAT"),
        "ectopic atrial tachycardia",
        regex=True,
        case=True,
    )

    texts = texts.str.replace(
        border_alphanumeric("AVR"), # Can't parse "AVR" as lowercase, since we may parse the "aVR" lead
        "accelerated ventricular rhythm",
        regex=True,
        case=True,
    )

    # Change 's into s for select acronyms, e.g., "PAC's" to "PACs"
    for acronym in ["PAC", "APB", "APC", "PVC", "VPB", "PSC", "SVPB", "PJC", "ECG"]:
        texts = texts.str.replace(
            border_alphanumeric(f"{acronym}'s"),
            f"{acronym}s",
            regex=True,
        )

    # TEMPORARY - TODO - Replace all sinus block patterns with "sinus block" so that we can assume Mobitz I/II refer to an AV block (until we can parse it better)
    texts = texts.str.replace("SA block", "sinus block", case=False, regex=False)
    texts = texts.str.replace("SA block (Mobitz I)", "sinus block", case=False, regex=False)
    texts = texts.str.replace("SA block (Mobitz II)", "sinus block", case=False, regex=False)
    texts = texts.str.replace("2nd degree SA block (Mobitz I)", "sinus block", case=False, regex=False)
    texts = texts.str.replace("2nd degree SA block (Mobitz I)", "sinus block", case=False, regex=False)
    texts = texts.str.replace("sinus exit block", "sinus block", case=False, regex=False)
    texts = texts.str.replace("sinus exit-block", "sinus block", case=False, regex=False)
    texts = texts.str.replace("2:1 SA exit block", "sinus block", case=False, regex=False)
    texts = texts.str.replace("sinoatrial block", "sinus block", case=False, regex=False)
    texts = texts.str.replace("sinoatrial exit block", "sinus block", case=False, regex=False)


    # Loss of capture/sensing
    # "failure to"
    texts = texts.str.replace("pacer failure to capture", "loss of capture", case=False, regex=False)
    texts = texts.str.replace("Failure to ventricular capture", "loss of ventricular capture", case=False, regex=False)
    texts = texts.str.replace("Failure to atrial capture", "loss of atrial capture", case=False, regex=False)
    texts = texts.str.replace("intermittent failure to capture", "intermittent loss of capture", case=False, regex=False)
    texts = texts.str.replace("with failure to capture", "loss of capture", case=False, regex=False)
    texts = texts.str.replace("(failure to capture)", "loss of capture", case=False, regex=False)
    texts = texts.str.replace("failure to capture", "loss of capture", case=False, regex=False)

    texts = texts.str.replace("pacemaker failure to sense/capture", "loss of sensing; loss of capture", case=False, regex=False)
    texts = texts.str.replace("Failure to sense and/or capture", "loss of sensing; loss of capture", case=False, regex=False)
    texts = texts.str.replace("Failure to sense / pace", "loss of sensing; loss of capture", case=False, regex=False)
    texts = texts.str.replace("failure to sense and capture by the pacemaker", "loss of sensing; loss of capture", case=False, regex=False)
    texts = texts.str.replace("failure to sense and capture", "loss of sensing; loss of capture", case=False, regex=False)
    texts = texts.str.replace("failure to sense with functional non-capture", "loss of sensing; loss of capture", case=False, regex=False)
    texts = texts.str.replace("Periodic failure to sense atrial activity as well as failure to pace atria", "loss of atrial sensing; loss of atrial capture", case=False, regex=False)

    texts = texts.str.replace("with failure to sense", "loss of sensing", case=False, regex=False)
    texts = texts.str.replace("Intermittant failure to sense", "intermittant loss of sensing", case=False, regex=False)
    texts = texts.str.replace("Pacer failure to sense", "loss of sensing", case=False, regex=False)
    texts = texts.str.replace("failure to sense intrinsic beats", "loss of sensing", case=False, regex=False)
    texts = texts.str.replace("failure to track P waves", "loss of atrial sensing", case=False, regex=False)
    texts = texts.str.replace("failure to sense", "loss of sensing", case=False, regex=False)

    # "failure of"
    texts = texts.str.replace("Failure of pacemaker capture", "loss of capture", case=False, regex=False)
    texts = texts.str.replace("Failure of ventricular capture", "loss of ventricular capture", case=False, regex=False)

    texts = texts.str.replace("Failure of pacemaker sensing and capture", "loss of sensing; loss of capture", case=False, regex=False)

    texts = texts.str.replace("Failure of sensing by pacemaker", "loss of sensing", case=False, regex=False)
    texts = texts.str.replace("failure of V sensing", "loss of ventricular sensing", case=False, regex=False)
    texts = texts.str.replace("Failure of sensing", "loss of sensing", case=False, regex=False)


    # Convert to lowercase - assumes all capitalization info has been preserved by now
    texts = texts.str.lower()

    # Turn diagnosis ending in "Normal" into "Normal ECG"
    normal = texts.str.endswith(" normal")
    texts[normal] = texts[normal] + " ecg"
    del normal

    # Turn diagnosis ending in "Abnormal" into "Abnormal ECG"
    abnormal = texts.str.endswith(" abnormal")
    texts[abnormal] = texts[abnormal] + " ecg"
    del abnormal

    # Turn diagnosis ending in "Borderline" into "Borderline ECG"
    borderline = texts.str.endswith(" borderline")
    texts[borderline] = texts[borderline] + " ecg"
    del borderline

    # Replace "`" with whitespace
    texts = texts.str.replace("`", " ", regex=False)
 
    # Replace '/' having whitespace on either side with just '/'
    # E.g., "atrial flutter / fibrillation" -> "atrial flutter/fibrillation"
    texts = texts.str.replace(r'\s*/\s*', '/', regex=True)

    # Handle different wave formats, matching:
    # - A single letter (any case) at a word boundary
    # - Optionally followed by a space, hyphen, and another space
    # - Ending either with "wave" or "waves".
    # Used to standardize expressions like "T-wave", "r -wave", "p -waves" to "T wave", "r wave", "p waves"
    texts = texts.str.replace(r"\b([a-zA-Z])\s?-?\s?wave(s?)\b", r"\1 wave\2", regex=True)

    # Fix typos like "Rwave" or "twave"
    texts = texts.str.replace(r"([A-Za-z])(wave)", r'\1 \2', regex=True)

    # Convert "&" to "and", accounting for poor spacing, e.g., "ST& T" and "ST &T"
    texts = texts.str.replace("&", " and ", regex=False).str.replace(r"\s+", " ", regex=True)

    # Fix before unbordered replacement of degree texts
    texts = texts.str.replace(
        border_alphanumeric("deghree"),
        "degree",
        regex=True,
        case=True,
    )
    texts = texts.str.replace(
        border_alphanumeric("deg"),
        "degree",
        regex=True,
        case=True,
    )

    return texts

def post_replacement(texts):
    texts = texts.str.replace('beat(s)', 'beats', regex=False)
    texts = texts.str.replace('for age/sex', '', regex=False)
    texts = texts.str.replace('are probably related to', 'conceivably related to', regex=False)
    texts = texts.str.replace('may be related to', 'conceivably related to', regex=False)
    texts = texts.str.replace('repeat if myocardial injury is suspected', '', regex=False)
    texts = texts.str.replace("interpretation made without knowing patient's age", '', regex=False)
    texts = texts.str.replace("interpretation made without knowledge of patient's age", '', regex=False)

    def replace_lead_specific(texts: pd.Series) -> pd.Series:
        texts = texts.copy()
    
        # Define the base pattern for a single lead
        base_pattern = r"\b(?:i{1,3}|avr|avl|avf|v[1-6])\b"
        run_pattern = rf"({base_pattern}(?:[ ,]*{base_pattern})*)"
        
        # Define contextual patterns for leads
        contextual_patterns = OrderedDict([
            ('lead <LEAD> unsuitable for analysis', rf"(lead\(s\) unsuitable for analysis: {run_pattern})"),
            ('missing lead <LEAD>', rf"(missing lead\(s\): {run_pattern})"),
            ('partial missing lead <LEAD>', rf"(partial missing lead\(s\): {run_pattern})"),
            ('artifact in lead <LEAD>', rf"(artifact in lead\(s\) {run_pattern})"),
            ('baseline wander in lead <LEAD>', rf"(baseline wander in lead\(s\) {run_pattern})"),
            ('lead <LEAD> not used for morphology analysis', rf"(lead\(s\) {run_pattern} were not used for morphology analysis)"),
            ('lead <LEAD> omitted due to possible sequence error', rf"(possible sequence error: {run_pattern} omitted)"),
            ('lead <LEAD> omitted due to sequence error', rf"(sequence error: {run_pattern} omitted)"),
        ])

        # Extract all contextual matches using str.extractall
        for repl, pattern in contextual_patterns.items():
            matches = texts.str.extractall(pattern).rename(columns={0: 'match', 1: 'leads'})
            leads_expl = matches['leads'].str.split('[ ,]', regex=True).explode()
            leads_expl = leads_expl[leads_expl != ''].copy()
            leads_expl = repl.split('<LEAD>')[0] + leads_expl + repl.split('<LEAD>')[1]
            leads_repl = leads_expl.groupby(level=0).apply('; '.join).str.strip()
            matches = matches.droplevel(1)
            matches['repl'] = leads_repl
        
            # Replace extracted so it can be pattern matched
            for index, row in matches.iterrows():
                texts.loc[index] = texts.loc[index].replace(row['match'], row['repl'])
        
        return texts

    texts = replace_lead_specific(texts)

    # Standardize location text
    combine_parts = [
        ("supra", "ventricular"),
        ("intra", "ventricular"),
        ("intra", "atrial"),
        ("inter", "atrial"),
    ]

    for part1, part2 in combine_parts:
        texts = texts.str.replace(f"{part1}- {part2}", part1 + part2, regex=False)
        texts = texts.str.replace(f"{part1} -{part2}", part1 + part2, regex=False)
        texts = texts.str.replace(f"{part1}-{part2}", part1 + part2, regex=False)
        texts = texts.str.replace(f"{part1} {part2}", part1 + part2, regex=False)

    # Remove "*" indicating importance (already parsed degrees, e.g., 1* AV block)
    texts = texts.str.replace("*", " ", regex=False)

    # Replace "v-pacing" with "ventricular pacing", "a-sensing" with "atrial sensing", etc.
    #  -   (?i) makes the pattern case-insensitive
    #  -   \b ensures word boundary, e.g., 'v-' is not directly preceded by alphanumeric character
    #  -   v- matches the literal 'v-' (case-insensitive due to the flag)
    #  -   \s* matches zero or more spaces
    #  -   (\w\w+) captures two or more word characters (the next word)
    texts = texts.str.replace(r'(?i)(\b)v-\s*(\w\w+)', r'\1ventricular \2', regex=True)
    texts = texts.str.replace(r'(?i)(\b)a-\s*(\w\w+)', r'\1atrial \2', regex=True)

    # Find instances of "bundle" not followed by "branch" and add in branch
    # e.g., "right bundle conduction delay" -> "right bundle branch conduction delay"
    texts = texts.str.replace(
        r'bundle(?! branch)',
        'bundle branch',
        case=False,
        regex=True,
    )

    # Standardize sensing/pacing hyphens
    texts = texts.str.replace("-sensing", " sensing", regex=False)
    texts = texts.str.replace("-pacing", " pacing", regex=False)

    # Perform ConjPattern replacements last
    CONJPAT_REPLACE_RESULTS["texts"] = texts
    for name, args in CONJPATS:
        texts_before = texts.copy()

        print(f"Replacing '{name}' conjunct patterns...")
        conjpat = ConjunctPatterns(*args, ignore_dups=True)
        CONJPAT_RESULTS[name] = conjpat()

        pat_repl_res = conjpat.replace(texts)
        CONJPAT_REPLACE_RESULTS[name] = pat_repl_res
        texts = pat_repl_res.replaced

        print(f"Performed replacements in {(texts != texts_before).sum()} entries.")

    texts = texts.str.strip()

    return texts

def preprocess_texts(texts):
    """
    Preprocess the texts.
    """
    # Changes before replacements
    print("Pre replacements...")
    texts = pre_replacement(texts)

    # Replacements - Unbordered, bordered, and regex
    print("Replacements...")
    print("Replacements - Unbordered")
    for old, new in dict_ordered_by_len(REPLACE["unbordered"]).items():
        texts = texts.str.replace(old, new, regex=False)

    print("Replacements - Bordered")
    texts, _ = replace(texts, REPLACE["bordered"], sort_by_length=False)

    print("Replacements - Regex")
    for old, new in REPLACE["regex"].items():
        texts = texts.str.replace(old, new, regex=True)

    # Changes after replacements
    print("Post replacements...")
    texts = post_replacement(texts)

    return texts
