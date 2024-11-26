from flask import Flask, render_template, request, redirect
import joblib
import sklearn

app = Flask(__name__)

# Load trained models
disease_model = joblib.load('D:\Code\Disease prediction drug recommendation\decision_tree.pkl')
drug_model = joblib.load('D:\Code\Disease prediction drug recommendation\medical_rf.pkl')

# Disease prediction mapping (Symptoms to Diseases)
# Additional entries for symptom_to_disease
symptom_to_disease = {
    'Fever': 'Flu',
    'Cough': 'Flu',
    'Headache': 'Migraine',
    'Fatigue': 'Anemia',
    'Joint pain': 'Arthritis',
    'Rash': 'Allergy',
    'Sore throat': 'Common cold',
    'Shortness of breath': 'Asthma',
    'Abdominal pain': 'Gastritis',
    'Nausea': 'Food poisoning'
}

# Additional entries for disease_to_drugs
disease_to_drugs = {
    'Flu': ['Tamiflu', 'Advil', 'Theraflu'],
    'Migraine': ['Excedrin', 'Ibuprofen', 'Maxalt'],
    'Anemia': ['Iron supplements', 'Vitamin B12', 'Folic acid'],
    'Arthritis': ['Ibuprofen', 'Prednisone', 'Celebrex'],
    'Allergy': ['Claritin', 'Zyrtec', 'Benadryl'],
    'Common cold': ['Tylenol Cold', 'Sudafed', 'Mucinex'],
    'Asthma': ['Albuterol', 'Advair', 'Singulair'],
    'Gastritis': ['Antacids', 'Proton pump inhibitors', 'H2 blockers'],
    'Food poisoning': ['Oral rehydration solution', 'Antidiarrheal medication', 'Anti-nausea medication']
}

# URL mapping for drugs
# Additional entries for drug_urls
drug_urls = {
    'Tamiflu': 'https://www.medplusmart.com/product/tamiflu-75mg-cap_tami0009',
    'Advil': 'https://www.ubuy.co.in/product/4DURJBQPO-advil-multi-symptom-cold-flu-pain-fever-reducer-50-ct',
    'Theraflu': 'https://www.ubuy.co.in/brand/theraflu',
    'Excedrin': 'https://www.ubuy.co.in/brand/excedrin',
    'Ibuprofen': 'https://dir.indiamart.com/chennai/ibuprofen.html',
    'Maxalt': 'https://pharmeasy.in/online-medicine-order/maxalt-rpd-10mg-tablet-93737',
    'Iron supplements': 'https://dir.indiamart.com/chennai/iron-tablet.html',
    'Vitamin B12': 'https://www.1mg.com/categories/fitness-supplements/health-food-drinks-6',
    'Folic acid': 'https://www.indiamart.com/proddetail/5mg-frefol-folic-acid-tablets-7119144862.html',
    'Prednisone': 'https://www.indiamart.com/proddetail/prednisone-tablets-17625677112.html',
    'Celebrex': 'https://www.indiamart.com/proddetail/celebrex-tablets-100-mg-worldwide-delivery-2853282380091.html',
    'Claritin': 'https://www.ubuy.co.in/brand/claritin',
    'Zyrtec': 'https://www.apollopharmacy.in/medicine/zyrtec-tablet-10mg',
    'Benadryl': 'https://pharmeasy.in/online-medicine-order/benadryl-25mg-capsule-37063',
    'Tylenol Cold': 'https://www.ubuy.co.in/product/5GVD2BC-tylenol-cold-flu-sever-size-24ct-tylenol-cold-flu-sever-24ct',
    'Sudafed': 'https://www.ubuy.co.in/product/5G71MIO-sudafed-pe-maximum-strength-congestion-sinus-pressure-relief-tablets-36ct',
    'Mucinex': 'https://www.netmeds.com/prescriptions/mucinex-tablet',
    'Albuterol': 'https://www.indiamart.com/proddetail/buy-albuterol-hfa-90-mcg-inhaler-online-2849776228662.html',
    'Advair': 'https://www.indiamart.com/proddetail/advair-diskus-salmeterol-fluticasone-500mcg-100-mcg-21790363733.html',
    'Singulair': 'https://pharmeasy.in/online-medicine-order/singulair-10mg-strip-of-15-tablets-20349',
    'Antacids': 'https://www.1mg.com/otc/digene-antacid-antigas-gel-for-acidity-gas-heartburn-bloated-stomach-relief-flavour-mint-otc220728',
    'Proton pump inhibitors': 'https://in.iherb.com/pr/life-extension-5-lox-inhibitor-with-apresflex-100-mg-60-vegetarian-capsules/40487?gad_source=1&gclid=CjwKCAiA0bWvBhBjEiwAtEsoWwB1F2DyTbudkN1eo-B8fONGpoHnETj0U8MEOYOur8Wa1Bq4PELdCRoCBM8QAvD_BwE&gclsrc=aw.ds',
    'H2 blockers': 'https://www.sova.health/products/pop-to-debloat-supplement-for-bloating-relief?variant=45126733824321&utm_source=Google&utm_medium=CPC&utm_campaign=Supplements+%7C+Sales+%7C+PMax+%7C+Shopping_Creatives+%7C+MXC+%7C+23rd+Feb&gad_source=1&gclid=CjwKCAiA0bWvBhBjEiwAtEsoW_S-8iWv_O0QwfJTAYgRh-1caRqk0hxFTlUsrcNWkrDWAXeySeclIhoCrLIQAvD_BwE',
    'Oral rehydration solution': 'https://www.fastandup.in/product/reload-combo-of-4-tubes-lime-lemon-flavour?gad_source=1&gclid=CjwKCAiA0bWvBhBjEiwAtEsoWxkzXMzX2qVk8BHqRA0S1ExLq41h99bdZgN8DnPkp2JM8ghgjw9FExoCMzMQAvD_BwE',
    'Antidiarrheal medication': 'https://www.leefordonline.in/product/312/leeford-pilogo-capsule?utm_source=googleads&vt_keyword=&vt_campaign=20633896036&vt_adgroup=159092384351&vt_loc_interest=&vt_physical=9061920&vt_matchtype=&vt_network=g&vt_placement=&gad_source=1',
    'Anti-nausea medication': 'https://www.medicinedirect.co.uk/travel-general-health/nausea/cyclizine-hydrochloride'
}

feedback_data = []

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        predicted_disease = predict_disease(symptoms)
        return redirect(f'/recommendations/{predicted_disease}')

def predict_disease(symptoms):
    for symptom in symptoms:
        if symptom in symptom_to_disease:
            return symptom_to_disease[symptom]
    return "Unknown"

@app.route('/recommendations/<disease>')
def recommendations(disease):
    drugs = disease_to_drugs.get(disease, [])
    return render_template('recommendations.html', disease=disease, drugs=drugs, get_drug_url=get_drug_url, feedback_data=feedback_data)

@app.route('/get_drug_url/<drug>')
def get_drug_url(drug):
    return drug_urls.get(drug, '#')  # If drug URL not found, return '#' as fallback

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        feedback = request.form['feedback']
        
        # Append the received feedback to the feedback_data list
        feedback_data.append({'name': name, 'email': email, 'feedback': feedback})
        
        return redirect('/')  # Redirect to homepage after feedback submission
if __name__ == '__main__':
    app.run(debug=True)
