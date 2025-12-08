from transformers import pipeline, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("./output_model", use_fast=False)
classifier = pipeline(
    "text-classification",
    model="./output_model",
    # tokenizer=tokenizer,
    truncation=True,
    max_length=512,
)

def data_preprocess(text):
    words = text.split()
    total_length = len(words)
    
    step_size = max(1, int(total_length * 0.1))  # 至少增加1
    
    current_end = step_size
    all_text = []
    while current_end <= total_length:
        window_words = words[0:current_end]
        
        window_text = " ".join(window_words)
        
        all_text.append(window_text)
        
        current_end += step_size
    
    # 确保最后一个窗口包含所有数据（如果还没包含的话）
    if current_end - step_size < total_length:
        window_words = words[0:total_length]
        window_text = " ".join(window_words)
        all_text.append(window_text)
    
    return all_text

def postprocess_formal(result, text_ori):
    all_text = data_preprocess(text_ori)
    label_trans = {"LABEL_0": 0, "LABEL_2": 1}
    current_label = result[0]['label']
    text_label = []
    for i, text in enumerate(all_text):
        if result[i]['label'] == current_label:
            text_label = [label_trans[current_label]]*len(text.split())
            if len(text_label) > len(text_ori.split()):
                text_label = [label_trans[current_label]]*len(text_ori.split())
        else:
            if current_label == "LABEL_0":
                next_label = "LABEL_2"
            elif current_label == "LABEL_2":
                next_label = "LABEL_0"
            extra_length = len(text_ori.split())-len(text_label)
            text_label.extend([label_trans[next_label]]*extra_length)
            return text_label
    return text_label

def postprocess(result, all_text):
    current_label = result[0]['label']
    sub_text = ""
    for i, text in enumerate(all_text):
        if result[i]['label'] == current_label:
            sub_text += text + " "
        else:
            return sub_text, current_label
    return sub_text, current_label

# single text
# text = "**Balancing Public Safety and Rehabilitation in Early Parole Decisions for Violent Offenses**  \n\n**Executive Summary**  \nThis report examines the criteria for early parole in cases involving violent offenses, focusing on the tension between public safety concerns and rehabilitation potential. Drawing on legislative frameworks, empirical studies, and risk assessment tools, the analysis identifies key factors influencing parole decisions, including age, offense severity, and rehabilitation metrics. The findings suggest that while age is a relevant factor, demonstrated rehabilitation and dynamic risk factors carry greater weight in parole evaluations. Safeguards such as structured risk assessments and post-release supervision are critical to ensuring transparency and fairness.  \n\n---  \n\n**Key Findings**  \n1. **Risk Assessment Tools**: Parole boards increasingly rely on validated instruments like the Level of Service Inventory (LSI-R) and COMPAS to evaluate recidivism risk, prioritizing dynamic factors (e.g., program completion, institutional behavior) over static factors (e.g., criminal history) [7][11].  \n2. **Age as a Factor**: While the \"age-crime curve\" suggests reduced criminal propensity after the mid-20s [9], age alone is not decisive. Older offenders with poor institutional behavior may still be deemed high-risk [11].  \n3. **Rehabilitation Metrics**: Demonstrated rehabilitation (e.g., therapy completion, vocational training) is the strongest predictor of parole suitability, accounting for ~45% of weighting in risk models [Python simulation].  \n4. **Violent Offenses**: These cases often trigger additional safeguards, such as mandatory supervision and victim input, to mitigate public safety risks [5][12].  \n\n---  \n\n**Evidence**  \n- **Legislative Frameworks**: California’s Proposition 57 emphasizes rehabilitation by expanding parole eligibility for non-violent offenders but maintains stricter scrutiny for violent crimes [1][2].  \n- **Empirical Data**: Meta-analyses show dynamic factors (e.g., rehabilitation progress) are more predictive of recidivism than static factors like age [7][11]."
# text = "nfants had severe global delay. However, approximately a third of infants showed mild or moderate delay in hearing and language, social or cognitive skill areas by 24 months. Developmental assessment undertaken by health visitors may be used to measure outcome in preterm infants. Severe developmental delay was at a level consistent with other follow-up studies of very preterm infants. Severe delay was identified by the 12-month check and was mainly in areas of motor function and language. High levels of mild to moderate developmental delay were identified at the 24-month"
# text = ", LV, EE)Belgique / BelgiëDeutschlandFranceNederlandSchweizUSAUnited KingdomInternational Choose your language Switch Call us: +1 888 988 5661 Contact us Book Classes My Account Checkout Shop Buy your bellicon® Buy accessories Ring ropes bellicon® accessories DVDs Introducing the bellicon® Which bungee strength? Which accessories are right for me? Training Academy Education bellicon Basic bellicon Move bellicon Bounce bellicon Circle Book Courses Exercise Classes Overview bellicon Move bellicon Bounce bellicon Circle bellicon Studio Chicago Studio Class Booking FAQs and Policy Careers Studio Blog Quality Best Built Best Bounce Compare Brands Expert Testimonials Customer Testimonials Media coverage Benefits bellicon® Basics Fitness benefits Overview Power Cardio Flexibility Balance Relaxation Health benefits Overview Back Pain Osteoporosis Lymph & Edema Osteoarthritis Weight Control Pelvic Floor Dysfunction Menopause Incontinence Pregnancy Postpartum Fibromyalgia Chronic Fatigue Syndrome Multiple Sclerosis Neurological Diseases Cancer Sports Medicine Rehabilitation Dancing & Performing Arts Senior Fitness Children's Classes About Us History Mission Statement Our Story Team The bellicon® Difference Bungee Technology Quality Control Safety Environmentally Friendly Research and Education Contact us Locations Chicago Studio Retail Locations News Press Releases Media Coverage Testimonials FAQs Terms & Conditions Privacy Policy Cookie policy Legal information © Bellicon LLC. All rights reserved. All product names are trademarks or registered trademarks of their respective owners. The bellicon® and bellicon Move®"
# text = "Floor Dysfunction Menopause Incontinence Pregnancy Postpartum Fibromyalgia Chronic Fatigue Syndrome Multiple Sclerosis Neurological Diseases Cancer Sports Medicine Rehabilitation Dancing & Performing Arts Senior Fitness Children's Classes About Us History Mission Statement Our Story Team The bellicon® Difference Bungee Technology Quality Control Safety Environmentally Friendly Research and Education Contact us Locations Chicago Studio Retail Locations News Press Releases Media Coverage Testimonials FAQs Terms & Conditions Privacy Policy Cookie policy Legal information © Bellicon LLC. All rights reserved. All product names are trademarks or registered trademarks of their respective owners. The bellicon® and bellicon Move®"
# text = ", LV, EE)Belgique / BelgiëDeutschlandFranceNederlandSchweizUSAUnited KingdomInternational Choose your language Switch Call us: +1 888 988 5661 Contact us Book Classes My Account Checkout Shop Buy your bellicon® Buy accessories Ring ropes bellicon® accessories DVDs Introducing the bellicon® Which bungee strength? Which accessories are right for me? Training Academy Education bellicon Basic bellicon Move bellicon Bounce bellicon Circle Book Courses Exercise Classes Overview bellicon Move bellicon Bounce bellicon Circle bellicon Studio Chicago Studio Class Booking FAQs and Policy Careers Studio Blog Quality Best Built Best Bounce Compare Brands Expert Testimonials Customer Testimonials Media coverage Benefits bellicon® Basics Fitness benefits Overview Power Cardio Flexibility Balance Relaxation Health benefits Overview Back Pain Osteoporosis Lymph & Edema Osteoarthritis Weight Control Pelvic Floor Dysfunction Menopause Incontinence Pregnancy Postpartum Fibromyalgia Chronic Fatigue Syndrome Multiple Sclerosis Neurological Diseases Cancer Sports Medicine Rehabilitation Dancing & Performing Arts Senior Fitness Children's Classes About Us History Mission Statement Our Story Team The bellicon® Difference Bungee Technology Quality Control Safety Environmentally Friendly Research and Education Contact us Locations Chicago Studio Retail Locations News Press Releases Media Coverage Testimonials FAQs Terms & Conditions Privacy Policy Cookie policy Legal information © Bellicon LLC. All rights reserved. All product names are trademarks or registered trademarks of their respective owners. The bellicon® and bellicon Move®"
# text = """ct { KeepCurrent, SetToNothing, } override OnInspectorGUI () { NGUIEditorTools.SetLabelWidth(120f); UIPlayTween tw = target as UIPlayTween; GUILayout.Space(6f); GUI.changed = false; GameObject tt = (GameObject)EditorGUILayout.ObjectField("Tween Target", tw.tweenTarget, typeof(GameObject), true); bool inc = EditorGUILayout.Toggle("Include Children", tw.includeChildren); int group = EditorGUILayout.IntField("Tween Group", tw.tweenGroup, GUILayout.Width(160f)); AnimationOrTween.Trigger trigger = (AnimationOrTween.Trigger)EditorGUILayout.EnumPopup("Trigger condition", tw.trigger); AnimationOrTween.Direction dir = (AnimationOrTween.Direction)EditorGUILayout.EnumPopup("Play direction", AnimationOrTween.EnableCondition enab = (AnimationOrTween.EnableCondition)EditorGUILayout.EnumPopup("If target is 2D, use the following to enable this tween.", tt.GetComponent<UIPanel>() != null ? AnimationOrTween.Enabled : tw.enable); if Vector3.zero)) EditorGUILayout.HelpBox(string.Format("{0} {1}", "This field will be ignored and replaced with the offset position from your play button. If you want to use this value instead of that one then just click on it.", NGUIEditorTools.Button("Use Offset Position")), MessageType.Info); if (!NGUIMath.IsVector3Equal(tw.offsetPositionAfterPlay, Vector3.zero)) EditorGUILayout.HelpBox(string.Format("{0} {1}", "This field will be ignored and replaced with the offset position from your play button. If you want to use this value instead of that one then just click on it.", NGUIEditorTools.Button("Use Offset Position")), MessageType.Info); if (!NGUIMath.IsVector3Equal(tw.offsetRotationBeforePlay, Vector3.zero)) EditorGUILayout.HelpBox(string.Format("{0} {1}", "This field will be ignored and replaced with the offset rotation from your play button. If you want to use this value instead of that one then just click on it.", NGUIEditorTools.Button("Use Offset Rotation")), MessageType.Info); if (!NGUIMath.IsVector3Equal(tw.offsetRotationAfterPlay, Vector3.zero)) EditorGUILayout.HelpBox(string.Format("{0} {1}", "This field will be ignored and replaced with the offset rotation from your play button. If you want"""
# text = """to be listing since we are using it by default. But when i do "/usr/sbin/apache2 -V | grep MPM" it shows "Server 1 worker thread(s)." So how can I know if its running on prefork or not? A: You should edit the /etc/httpd/conf.modules.d/00-mpm.conf file and comment out PreforkModule line, then restart Apache. Q: Apache enable mod_fcgid I have in"""
text = """interest. ButMehl found recognition among supporters of naturalhealing methods whosaw their principles confirmed by his discovery. Mehl's tentative therapy was superseded by technologicalmethods of sunlight treatment, as developed by the Danish doctor Niels Ryberg Finsen. In 1903 Finsen wasawarded the NobelPrize for hiswork while Mehl's discovery remained practically unknown. This episode inthehistory of medicine demonstrates how the recognition of"""
# text_l = text.split()
# print(len(text_l))
# exit(0)
result = classifier(data_preprocess(text))
# result = classifier(text)

print(result)
sub_text, current_label = postprocess(result, data_preprocess(text))
print(sub_text)
print(current_label)

text_label = postprocess_formal(result, text)
print(text_label)

print(len(text_label))
print(len(text.split()))