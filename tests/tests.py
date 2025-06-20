import pytest
from pathlib import Path
import sys

# Path standardisation
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
try:
    from src.backend.text_classification import Classifier
except ImportError:
    print("Warning: Could not import 'Classifier' from 'src.backend.text_classification'.")
    print("Please ensure 'src/backend/text_classification.py' exists and is correctly configured.")
    print("The classification step will be skipped if Classifier is not available.")
    Classifier = None # Set to None if import fails

fox_article = """
FBI Director Kash Patel told --- that the case involving two Chinese nationals who were charged Tuesday with allegedly smuggling a "dangerous biological pathogen" into the U.S. to study at the University of Michigan laboratory demonstrates a serious national security threat to America's food supply. 

"This case is a sobering reminder that the Chinese Communist Party continues to deploy operatives and researchers to infiltrate our institutions and target our food supply, an act that could cripple our economy and endanger American lives," Patel told --- in a statement. "Smuggling a known agroterrorism agent into the U.S. is not just a violation of law, it’s a direct threat to national security. I commend the FBI Detroit Division and our partners at CBP for stopping this biological threat before it could do real damage."

University of Michigan research fellow Yunqing Jian and her boyfriend, Zunyong Liu – both citizens of the People's Republic of China – have been charged in a criminal complaint with conspiracy, smuggling goods into the United States, false statements, and visa fraud, the U.S. Attorney for the Eastern District of Michigan announced Tuesday. The investigation is being conducted by the FBI and U.S. Customs and Border Protection (CBP). 

The couple are accused of smuggling a fungus called Fusarium graminearum, which scientific literature classifies as a "potential agroterrorism weapon," according to the Justice Department. Federal prosecutors note the noxious fungus causes "head blight," a disease of wheat, barley, maize, and rice, and "is responsible for billions of dollars in economic losses worldwide each year." 

JUDGE TO BLOCK TRUMP ADMIN'S HARVARD FOREIGN STUDENTS BAN
Chinese flag imposed on University of Michigan campus scene

Two Chinese nationals were charged with conspiracy and smuggling a "dangerous biological pathogen" into the U.S. for their work at a University of Michigan laboratory. (Detroit Free Press/IMAGN)

The Justice Department also says fusarium graminearum’s toxins cause vomiting, liver damage, and "reproductive defects in humans and livestock." 

According to the criminal complaint, Jian, 33, allegedly received Chinese government funding for her work on the pathogen in China. 

Federal prosecutors say Jian’s electronics contain information "describing her membership in and loyalty to the Chinese Communist Party." 
"""

cnn_article = """
From day to day, Donald Trump’s second term often seems like a roman candle of grievance, with the administration spraying attacks in all directions on institutions and individuals the president considers hostile.

Hardly a day goes by without Trump pressuring some new target: escalating his campaign against Harvard by trying to bar the university from enrolling foreign students; deriding musicians Bruce Springsteen and Taylor Swift on social media; and issuing barely veiled threats against Walmart and Apple around the companies’ responses to his tariffs.

Trump’s panoramic belligerence may appear as to lack a more powerful unifying theme than lashing out at anything, or anyone, who has caught his eye. But to many experts, the confrontations Trump has instigated since returning to the White House are all directed toward a common, and audacious, goal: undermining the separation of powers that represents a foundational principle of the Constitution.

While debates about the proper boundaries of presidential authority have persisted for generations, many historians and constitutional experts believe Trump’s attempt to centralize power over American life differs from his predecessors’ not only in degree, but in kind.
American flags are seen during a protest outside the US Supreme Court over President Donald Trump's move to end birthright citizenship as the court hears arguments over the order in Washington, DC, on May 15.
American flags are seen during a protest outside the US Supreme Court over President Donald Trump's move to end birthright citizenship as the court hears arguments over the order in Washington, DC, on May 15.
Drew Angerer/AFP/Getty Images

At various points in our history, presidents have pursued individual aspects of Trump’s blueprint for maximizing presidential clout. But none have combined Trump’s determination to sideline Congress; circumvent the courts; enforce untrammeled control over the executive branch; and mobilize the full might of the federal government against all those he considers impediments to his plans: state and local governments and elements of civil society such as law firms, universities and nonprofit groups, and even private individuals.

“The sheer level of aggression and the speed at which (the administration has) moved ” is unprecedented, said Paul Pierson, a political scientist at the University of California at Berkeley. “They are engaging in a whole range of behaviors that I think are clearly breaking through conventional understandings of what the law says, and of what the Constitution says.”

Yuval Levin, director of social, cultural and constitutional studies at the conservative American Enterprise Institute, also believes that Trump is pursuing the most expansive vision of presidential power since Woodrow Wilson over a century ago.

But Levin believes Trump’s campaign will backfire by compelling the Supreme Court to resist his excesses and more explicitly limit presidential authority. “I think it is likely that the presidency as an institution will emerge from these four years weaker and not stronger,” Levin wrote in an email. “The reaction that Trump’s excessive assertiveness will draw from the Court will backfire against the executive branch in the long run.”

Other analysts, to put it mildly, are less optimistic that this Supreme Court, with its six-member Republican-appointed majority, will stop Trump from augmenting his power to the point of destabilizing the constitutional system. It remains uncertain whether any institution in the intricate political system that the nation’s founders devised can do so.
A war on multiple fronts

One defining characteristic of Trump’s second term is that he’s moving simultaneously against all of the checks and balances the Constitution established to constrain the arbitrary exercise of presidential power.

"""

# Define a fixture for the Classifier
@pytest.fixture(scope="session")
def classifier_instance():
    return Classifier()

def test_classifier_initialization_success(classifier_instance):
    # Checking if initialization completes without error
    assert classifier_instance.tok is not None
    assert classifier_instance.model is not None
    assert classifier_instance.id2label is not None
    assert isinstance(classifier_instance.id2label, dict)

def test_classify_basic_text(classifier_instance):

    # Note: expected label could differ for different models
    label_fox = classifier_instance.classify(fox_article, return_probs=False)
    assert isinstance(label_fox, str)
    assert label_fox == 'left'
    print(f"Assertion passed: label_fox is '{label_fox}' (expected 'left-center')")

    label_cnn = classifier_instance.classify(cnn_article, return_probs=False)
    assert isinstance(label_cnn, str)
    assert label_cnn == 'left'
    print(f"Assertion passed: label_fox is '{label_cnn}' (expected 'left')")

def test_classify_with_probabilities(classifier_instance):
    text = fox_article
    label, probs = classifier_instance.classify(text, return_probs=True)
    assert isinstance(label, str)
    assert isinstance(probs, dict)
    assert all(isinstance(k, str) for k in probs.keys())
    assert all(isinstance(v, float) for v in probs.values()) # Crucial for JSON serialization
    assert all(0.0 <= v <= 1.0 for v in probs.values()) # Probabilities should be between 0 and 1
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3) # Sum of probs is ~1
