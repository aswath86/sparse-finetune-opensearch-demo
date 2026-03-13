#!/usr/bin/env python3
"""
Step 0: Index domain documents into OpenSearch.

Creates a 'health-articles' index with 49 documents covering respiratory
illness, vaccination, infection control, and public health. Represents the
"existing domain data" in your cluster.

Usage:
    python index_data.py
"""
import json, urllib.request

OS_URL = "http://localhost:9202"

ARTICLES = [
    # --- From covid_train.jsonl positives (rewritten as articles) ---
    {"title": "Loss of Smell and Taste in Respiratory Infections",
     "content": "Loss of smell and taste is a hallmark of respiratory virus infection. Inflammation of the olfactory nerve causes chronic sensory dysfunction. Fever and cough may accompany the infection. Patients report distorted or absent smell lasting weeks to months after initial symptoms resolve.",
     "category": "symptoms", "source": "National Health Review"},
    {"title": "Sudden Anosmia as an Early Indicator",
     "content": "Sudden loss of smell and taste indicates respiratory virus infection with inflammation of sensory nerves. Fever, fatigue, and cough are common accompanying symptoms. Early recognition of anosmia can prompt timely testing and isolation to prevent further transmission.",
     "category": "symptoms", "source": "Clinical Infectious Disease Journal"},
    {"title": "Parosmia Following Respiratory Illness",
     "content": "Distorted smell following respiratory infection indicates virus damage to olfactory receptors. Chronic inflammation delays recovery. Immunity develops after infection. Rehabilitation through olfactory training with essential oils shows promise in accelerating sensory recovery.",
     "category": "symptoms", "source": "Neurology Today"},
    {"title": "Airborne Transmission and Mask Effectiveness",
     "content": "Respiratory virus transmission occurs through airborne spread in crowded settings. Mask use reduces transmission risk during epidemic and outbreak periods. Vaccine and immunity provide additional protection. Well-fitted N95 respirators offer superior filtration compared to surgical masks.",
     "category": "prevention", "source": "WHO Technical Brief"},
    {"title": "Indoor Ventilation and Infection Control",
     "content": "Indoor mask use prevents respiratory virus transmission during epidemic outbreaks. Virus spread through respiratory droplets increases in enclosed spaces. Vaccine immunity reduces infection severity. Improved ventilation with HEPA filtration significantly lowers airborne virus concentration.",
     "category": "prevention", "source": "CDC Guidance Document"},
    {"title": "How Masks Filter Respiratory Particles",
     "content": "Masks filter respiratory virus particles and reduce transmission of infection. During epidemic outbreaks, mask use combined with vaccine immunity significantly lowers infection and mortality rates. Multi-layer cloth masks provide moderate protection when medical-grade masks are unavailable.",
     "category": "prevention", "source": "Journal of Aerosol Science"},
    {"title": "Booster Vaccination and Waning Immunity",
     "content": "Booster vaccine doses restore immunity against virus variants. Antibody levels decline over time after initial vaccine series. Booster vaccination reduces infection severity and mortality. Updated formulations targeting circulating variants provide broader cross-protection.",
     "category": "vaccination", "source": "Vaccine Research Quarterly"},
    {"title": "Maintaining Antibody Levels Through Boosters",
     "content": "Additional vaccine booster doses maintain antibody levels and immunity against new virus variants. Vaccine effectiveness against infection and mortality decreases without booster doses. Hybrid immunity from both vaccination and prior infection provides the most durable protection.",
     "category": "vaccination", "source": "Immunology Frontiers"},
    {"title": "Duration of Vaccine Protection",
     "content": "Vaccine immunity and antibody protection wane over months. Booster doses restore immunity against virus infection. New variants may reduce vaccine effectiveness requiring updated booster formulations. Cellular immunity persists longer than antibody levels and prevents severe disease.",
     "category": "vaccination", "source": "New England Journal of Medicine"},
    {"title": "Superspreader Events and Social Gatherings",
     "content": "Rapid spread of respiratory virus infection through social gatherings indicates high transmission. Fever, cough, and fatigue are common symptoms. Outbreak investigation traces infection transmission chains. Indoor events with poor ventilation pose the highest risk for cluster infections.",
     "category": "transmission", "source": "Epidemiology Weekly Report"},
    {"title": "Workplace Outbreaks and Containment",
     "content": "Workplace virus outbreak indicates respiratory infection transmission in enclosed settings. Epidemic spread causes widespread fever, cough, and fatigue. Isolation reduces further transmission. Employer-mandated ventilation improvements and sick leave policies reduce workplace outbreak severity.",
     "category": "transmission", "source": "Occupational Health Bulletin"},
    {"title": "Mechanisms of Respiratory Virus Spread",
     "content": "Respiratory virus transmission occurs through airborne particles during close social contact. Infection spreads rapidly during epidemic outbreaks. Mask use, isolation, and vaccine immunity reduce transmission. Fomite transmission plays a minor role compared to aerosol and droplet routes.",
     "category": "transmission", "source": "Lancet Infectious Diseases"},
    {"title": "Fever and Dry Cough: When to Seek Care",
     "content": "Fever with dry cough is characteristic of respiratory virus infection. Pneumonia and lung inflammation develop in severe cases. Oxygen support and respiratory monitoring are essential. Persistent fever above 39°C for more than three days warrants emergency evaluation.",
     "category": "symptoms", "source": "Emergency Medicine Guidelines"},
    {"title": "Respiratory Distress and Pneumonia",
     "content": "Cough with respiratory difficulty indicates virus infection with lung involvement. Pneumonia and pulmonary inflammation cause oxygen levels to drop. Respiratory support may be needed. Prone positioning improves oxygenation in patients with moderate to severe respiratory compromise.",
     "category": "treatment", "source": "Critical Care Medicine"},
    {"title": "Post-Infection Fatigue Syndrome",
     "content": "Chronic fatigue following virus infection persists for weeks or months. Inflammation and immune dysfunction cause prolonged fatigue. Cardiac and pulmonary complications contribute to ongoing exhaustion. Graded exercise therapy and pacing strategies help manage persistent fatigue symptoms.",
     "category": "long-term", "source": "Journal of Chronic Illness"},
    {"title": "Long-Term Neurological Effects of Infection",
     "content": "Chronic fatigue after respiratory virus infection involves persistent inflammation and immune dysfunction. Cardiac, pulmonary, and neurological complications cause prolonged fatigue and reduced exercise tolerance. Brain fog and concentration difficulties affect up to 30% of recovered patients.",
     "category": "long-term", "source": "Neuroscience Research Letters"},
    {"title": "Understanding Long-Term Post-Viral Illness",
     "content": "Chronic symptoms persisting after respiratory virus infection include fatigue, cough, and respiratory difficulty. Lung inflammation, cardiac complications, and immune dysfunction cause prolonged illness and mortality risk. Multidisciplinary rehabilitation programs improve functional outcomes.",
     "category": "long-term", "source": "British Medical Journal"},
    {"title": "Recognizing Early Symptoms of Infection",
     "content": "Respiratory virus infection presents with fever, cough, fatigue, and loss of smell. Antigen testing detects active infection. Antibody testing confirms prior immunity. Rapid point-of-care tests enable same-day diagnosis and treatment initiation in outpatient settings.",
     "category": "diagnosis", "source": "Diagnostic Medicine Review"},
    {"title": "Post-Exposure Testing Protocols",
     "content": "Testing after virus exposure detects respiratory infection before symptom onset. Antigen tests identify active infection. Isolation after positive results prevents transmission during the outbreak. Serial testing on days 3 and 5 post-exposure maximizes detection sensitivity.",
     "category": "diagnosis", "source": "Public Health Laboratory Network"},
    # --- Additional realistic articles ---
    {"title": "Pulse Oximetry for Home Monitoring",
     "content": "Home pulse oximetry enables early detection of silent hypoxia in respiratory virus patients. Oxygen saturation below 94% warrants medical evaluation. Continuous monitoring identifies rapid deterioration before symptoms worsen. Affordable fingertip devices provide reliable readings for most patients.",
     "category": "treatment", "source": "Telemedicine Journal"},
    {"title": "Antiviral Treatments for Respiratory Infection",
     "content": "Early antiviral treatment within 48 hours of symptom onset reduces virus replication and disease severity. Oral antiviral medications decrease hospitalization risk by 80% in high-risk patients. Treatment effectiveness diminishes significantly when initiated after the first five days of illness.",
     "category": "treatment", "source": "Pharmacology and Therapeutics"},
    {"title": "Corticosteroid Use in Severe Respiratory Disease",
     "content": "Corticosteroid therapy reduces mortality in patients requiring oxygen support for severe respiratory virus infection. Dexamethasone administered for ten days improves survival in hospitalized patients with pneumonia. Steroids are not recommended for mild cases without oxygen requirement.",
     "category": "treatment", "source": "Critical Care Medicine"},
    {"title": "Pediatric Respiratory Infections",
     "content": "Children with respiratory virus infection typically present with mild fever, runny nose, and cough. Severe cases may develop bronchiolitis or pneumonia requiring hospitalization. Multisystem inflammatory syndrome is a rare but serious complication occurring weeks after initial infection.",
     "category": "symptoms", "source": "Pediatrics International"},
    {"title": "Pregnancy and Respiratory Virus Risk",
     "content": "Pregnant women face elevated risk of severe respiratory virus infection and pneumonia. Vaccination during pregnancy provides protective antibody transfer to the newborn. Fever management with acetaminophen is recommended. Preterm delivery risk increases with severe maternal infection.",
     "category": "risk-factors", "source": "Obstetrics and Gynecology Review"},
    {"title": "Diabetes and Infection Severity",
     "content": "Patients with diabetes mellitus experience higher rates of severe respiratory virus infection and mortality. Hyperglycemia impairs immune function and increases inflammation. Tight glucose control during acute infection improves clinical outcomes. Vaccination is strongly recommended for all diabetic patients.",
     "category": "risk-factors", "source": "Diabetes Care Journal"},
    {"title": "Cardiovascular Complications of Viral Infection",
     "content": "Respiratory virus infection triggers cardiac inflammation, arrhythmias, and myocardial injury. Troponin elevation indicates cardiac involvement and predicts worse outcomes. Post-infection cardiac screening is recommended for patients with persistent chest pain or exercise intolerance.",
     "category": "complications", "source": "Cardiology Today"},
    {"title": "Mental Health Impact of Pandemic Isolation",
     "content": "Prolonged isolation during epidemic outbreaks increases rates of depression, anxiety, and substance abuse. Healthcare workers face elevated burnout and post-traumatic stress. Community mental health programs and telehealth counseling services help mitigate psychological harm during outbreaks.",
     "category": "public-health", "source": "Psychiatry Research"},
    {"title": "Contact Tracing and Outbreak Investigation",
     "content": "Systematic contact tracing identifies exposed individuals before symptom onset, enabling early isolation and testing. Digital contact tracing applications supplement manual investigation efforts. Rapid notification within 24 hours of positive test results maximizes transmission prevention effectiveness.",
     "category": "public-health", "source": "Epidemiology Weekly Report"},
    {"title": "Wastewater Surveillance for Early Detection",
     "content": "Monitoring virus RNA levels in municipal wastewater provides early warning of community transmission trends. Wastewater surveillance detects infection surges 7-10 days before clinical case counts rise. This population-level monitoring complements individual diagnostic testing programs.",
     "category": "public-health", "source": "Environmental Health Perspectives"},
    {"title": "Immune Response and T-Cell Memory",
     "content": "T-cell immunity provides durable protection against severe respiratory virus disease even as antibody levels decline. Memory T-cells recognize conserved virus proteins across multiple variants. Hybrid immunity from vaccination plus natural infection generates the broadest T-cell response.",
     "category": "immunology", "source": "Nature Immunology"},
    {"title": "mRNA Vaccine Technology and Development",
     "content": "Messenger RNA vaccines encode virus spike protein to stimulate immune response without using live virus. Rapid manufacturing enables updated formulations targeting new variants within weeks. Lipid nanoparticle delivery systems protect fragile mRNA molecules and facilitate cellular uptake.",
     "category": "vaccination", "source": "Science Translational Medicine"},
    {"title": "Vaccine Hesitancy and Public Communication",
     "content": "Vaccine hesitancy driven by misinformation reduces population immunity and prolongs epidemic outbreaks. Transparent communication about vaccine safety data and adverse event monitoring builds public trust. Community health workers and trusted local leaders are effective messengers for vaccine promotion.",
     "category": "public-health", "source": "WHO Policy Brief"},
    {"title": "Ventilator Management in Acute Respiratory Failure",
     "content": "Mechanical ventilation supports gas exchange in patients with severe pneumonia and acute respiratory failure. Lung-protective ventilation strategies with low tidal volumes reduce ventilator-induced injury. Early prone positioning for 16 hours daily improves oxygenation and survival rates.",
     "category": "treatment", "source": "Intensive Care Medicine"},
    {"title": "Monoclonal Antibody Therapy for High-Risk Patients",
     "content": "Monoclonal antibody infusions neutralize respiratory virus and prevent progression to severe disease in high-risk patients. Treatment must be administered within seven days of symptom onset for maximum effectiveness. Virus mutations may reduce antibody binding, requiring updated therapeutic formulations.",
     "category": "treatment", "source": "Journal of Infectious Diseases"},
    {"title": "School Reopening and Infection Mitigation",
     "content": "Safe school reopening during epidemic periods requires layered mitigation including ventilation improvement, mask policies, and regular testing. Vaccination of eligible students and staff reduces transmission. Outdoor learning spaces and cohort-based scheduling minimize close contact exposure.",
     "category": "public-health", "source": "Education and Health Policy"},
    {"title": "Air Travel and Respiratory Infection Risk",
     "content": "Aircraft cabin ventilation with HEPA filtration reduces but does not eliminate respiratory virus transmission during flights. Mask use during boarding, deplaning, and in-flight service periods provides additional protection. Pre-departure testing requirements reduce the probability of infectious passengers boarding.",
     "category": "prevention", "source": "Travel Medicine and Infectious Disease"},
    {"title": "Reinfection Risk and Variant Immune Escape",
     "content": "Prior respiratory virus infection provides partial immunity but does not prevent reinfection with antigenically distinct variants. Reinfection typically causes milder illness due to residual immune memory. Variant-specific mutations in the spike protein enable partial escape from neutralizing antibodies.",
     "category": "immunology", "source": "Virology Journal"},
    {"title": "Rapid Antigen Test Accuracy and Limitations",
     "content": "Rapid antigen tests detect active respiratory virus infection with high specificity but moderate sensitivity. False-negative results occur more frequently during early infection and in asymptomatic individuals. Serial testing on consecutive days improves overall detection accuracy for screening programs.",
     "category": "diagnosis", "source": "Clinical Chemistry"},
    {"title": "Oxygen Therapy Escalation Pathways",
     "content": "Progressive respiratory failure from virus pneumonia requires stepwise oxygen therapy escalation from nasal cannula to high-flow oxygen to non-invasive ventilation. Early recognition of deterioration using standardized early warning scores enables timely intervention and reduces emergency intubation rates.",
     "category": "treatment", "source": "Respiratory Care"},
    {"title": "Global Vaccine Distribution and Equity",
     "content": "Equitable global vaccine distribution remains critical for controlling respiratory virus transmission and preventing emergence of new variants. Low-income countries face significant barriers to vaccine access including cold chain infrastructure limitations and healthcare workforce shortages.",
     "category": "public-health", "source": "Global Health Policy Forum"},
    {"title": "Inflammatory Biomarkers in Severe Infection",
     "content": "Elevated C-reactive protein, ferritin, and interleukin-6 levels predict progression to severe respiratory virus disease. Serial biomarker monitoring guides treatment escalation decisions in hospitalized patients. Anti-inflammatory therapies targeting specific cytokine pathways reduce organ damage in severe cases.",
     "category": "diagnosis", "source": "Laboratory Medicine"},
    {"title": "Rehabilitation After Severe Respiratory Illness",
     "content": "Pulmonary rehabilitation programs improve exercise capacity and quality of life in patients recovering from severe respiratory virus pneumonia. Structured breathing exercises, progressive aerobic training, and psychological support address the multifaceted nature of post-infection disability.",
     "category": "long-term", "source": "Physical Therapy Journal"},
    {"title": "Coagulation Disorders in Viral Infection",
     "content": "Respiratory virus infection activates coagulation pathways leading to increased risk of deep vein thrombosis and pulmonary embolism. Prophylactic anticoagulation reduces thrombotic events in hospitalized patients. D-dimer elevation serves as a prognostic marker for disease severity and thrombotic risk.",
     "category": "complications", "source": "Thrombosis Research"},
    {"title": "Mucosal Immunity and Nasal Vaccines",
     "content": "Intranasal vaccine delivery stimulates mucosal immunity at the primary site of respiratory virus entry. Secretory IgA antibodies in nasal passages neutralize virus before systemic infection occurs. Nasal vaccines may provide superior protection against transmission compared to intramuscular injection.",
     "category": "vaccination", "source": "Mucosal Immunology"},
    {"title": "Genomic Surveillance and Variant Tracking",
     "content": "Whole genome sequencing of respiratory virus samples enables real-time tracking of variant emergence and spread. Genomic surveillance networks detect mutations affecting transmissibility, immune evasion, and treatment resistance. International data sharing through open databases accelerates global public health response.",
     "category": "public-health", "source": "Genomics and Public Health"},
    {"title": "Acute Kidney Injury in Hospitalized Patients",
     "content": "Severe respiratory virus infection causes acute kidney injury through direct viral damage, inflammation, and hemodynamic instability. Kidney involvement occurs in approximately 20% of hospitalized patients and significantly increases mortality risk. Early fluid management and avoidance of nephrotoxic medications are essential.",
     "category": "complications", "source": "Nephrology Dialysis Transplantation"},
    {"title": "Pediatric Vaccination Strategies",
     "content": "Vaccination of children against respiratory virus reduces household transmission and protects vulnerable family members. Age-appropriate dosing schedules ensure adequate immune response in developing immune systems. School-based vaccination programs achieve high coverage rates and reduce community transmission.",
     "category": "vaccination", "source": "Pediatric Infectious Disease Journal"},
    {"title": "Healthcare Worker Protection and PPE",
     "content": "Healthcare workers face elevated respiratory virus exposure during aerosol-generating procedures including intubation and bronchoscopy. Proper personal protective equipment including N95 respirators, face shields, and gowns reduces occupational infection risk. Fit testing and training ensure effective PPE utilization.",
     "category": "prevention", "source": "Infection Control and Hospital Epidemiology"},
    {"title": "Gastrointestinal Manifestations of Respiratory Virus",
     "content": "Respiratory virus infection causes gastrointestinal symptoms including nausea, diarrhea, and abdominal pain in approximately 15% of patients. Virus RNA detected in stool samples suggests potential fecal-oral transmission route. Gastrointestinal symptoms may precede respiratory manifestations in some patients.",
     "category": "symptoms", "source": "Gastroenterology"},
]


def main():
    # Delete index if exists
    try:
        req = urllib.request.Request(f"{OS_URL}/health-articles", method="DELETE")
        urllib.request.urlopen(req)
    except Exception:
        pass

    # Create index with realistic mapping
    mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "category": {"type": "keyword"},
                "source": {"type": "keyword"},
            }
        }
    }
    req = urllib.request.Request(f"{OS_URL}/health-articles",
        data=json.dumps(mapping).encode(), headers={"Content-Type": "application/json"}, method="PUT")
    urllib.request.urlopen(req)

    # Bulk index
    bulk = ""
    for i, doc in enumerate(ARTICLES):
        bulk += json.dumps({"index": {"_index": "health-articles", "_id": str(i+1)}}) + "\n"
        bulk += json.dumps(doc) + "\n"

    req = urllib.request.Request(f"{OS_URL}/_bulk",
        data=bulk.encode(), headers={"Content-Type": "application/x-ndjson"})
    resp = json.loads(urllib.request.urlopen(req).read())
    errors = sum(1 for item in resp["items"] if item["index"].get("error"))
    print(f"Indexed {len(ARTICLES)} documents ({errors} errors)")

    # Verify
    urllib.request.urlopen(urllib.request.Request(f"{OS_URL}/health-articles/_refresh", method="POST"))
    resp = json.loads(urllib.request.urlopen(f"{OS_URL}/health-articles/_count").read())
    print(f"Index count: {resp['count']}")

    # Show category breakdown
    agg_q = {"size": 0, "aggs": {"cats": {"terms": {"field": "category", "size": 20}}}}
    req = urllib.request.Request(f"{OS_URL}/health-articles/_search",
        data=json.dumps(agg_q).encode(), headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    print("Categories:")
    for b in resp["aggregations"]["cats"]["buckets"]:
        print(f"  {b['key']:20s} {b['doc_count']}")


if __name__ == "__main__":
    main()
