import json, random, copy, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load base schema
with open(os.path.join(BASE_DIR, "source_uschema.json"), "r") as f:
    base_model = json.load(f)

# Possible synonyms and variations
entity_synonyms = {
    "patient": ["patient", "person", "subject", "individual", "case"],
    "caregivers": ["caregiver", "staff", "nurse", "doctor", "assistant"],
    "Admission": ["admission", "visit", "hospitalization", "entry", "checkin"],
    "Prescription": ["prescription", "medication", "treatment", "drug_order", "therapy"],
    "Imaging": ["imaging", "radiology", "scan", "xray", "mri"],
    "AiAnnotation": ["annotation", "ai_result", "finding", "auto_label", "detection"],
    "Address": ["address", "location", "residence", "place", "geo"],
    "Contact": ["contact", "communication", "info", "coordination", "phone_info"],
    "Insurance": ["insurance", "coverage", "policy", "benefits", "plan"],
    "EmergencyContact": ["emergency_contact", "emergency_info", "ec_info", "urgent_contact", "emergency_person"],
    "NextOfKin": ["next_of_kin", "nok", "relative", "family_contact", "kin_info"],
    "Phone": ["phone", "telephone", "mobile", "cell", "contact_number"],
    "Email": ["email", "electronic_mail", "mail_address", "email_info", "contact_email"],
    "Policy": ["policy", "insurance_policy", "coverage_plan", "benefit_plan", "insurance_info"]
}

attr_synonyms = {
    "patients.name": ["fullname", "person_name", "pname", "identity"],
    "patients.gender": ["sex", "biological_sex"],
    "patients.dob": ["birth_date", "date_of_birth"],
    "patients.deceased": ["death_status", "is_dead", "alive_flag"],
    "admissions.admission_id": ["visit_id", "entry_id", "session_id"],
    "admissions.admit_time": ["entry_time", "start_time", "arrival_time"],
    "admissions.discharge_time": ["release_time", "end_time"],
    "admissions.telemedicine_flag": ["remote_visit", "virtual_flag"],
    "admissions.visit_mode": ["mode", "service_type"],
    "medications.drug": ["medication", "substance", "compound"],
    "medications.dose": ["dosage", "quantity"],
    "medications.route": ["administration_route", "pathway"],
    "medications.frequency": ["freq", "schedule"],
    "medications.modality": ["technique", "device_type"],
    "medications.body_site": ["body_part", "region", "scan_zone"],
    "caregivers.label": ["tag", "category", "classification"],
    "confidence": ["certainty", "probability", "assurance"],
    "value": ["data_value", "measurement", "reading"],
    "address_line": ["street_address", "location_line"],
    "phone_number": ["contact_number", "telephone_number", "mobile_number"],
    "email_address": ["email", "mail_address", "contact_email"],
    "policy_number": ["policy_id", "insurance_id"],
    "provider": ["insurer", "insurance_provider"],
    "coverage_type": ["plan_type", "benefit_type"],
    "relationship": ["relation", "connection", "link"],
}

types = ["String", "Number", "Boolean", "Null|String", "Array<String>"]

def random_synonym(word_dict, word):
    return random.choice(word_dict.get(word, [word]))

def mutate_attribute(attr):
    attr = copy.deepcopy(attr)
    if isinstance(attr, dict):
        if "name" in attr:
            attr["name"] = random_synonym(attr_synonyms, attr["name"])
        if "type" in attr:
            attr["type"] = random.choice(types)
    return attr

def mutate_entity(entity):
    entity = copy.deepcopy(entity)
    entity_type = entity["EntityType"]
    entity_type["name"] = random_synonym(entity_synonyms, entity_type["name"])

    for variation in entity_type.get("variations", []):
        sv = variation["StructuralVariation"]
        new_props = []
        for prop in sv.get("properties", []):
            if "Attribute" in prop:
                attrs = prop["Attribute"]
                if isinstance(attrs, list):
                    mutated_attrs = [mutate_attribute(a) for a in attrs]
                elif isinstance(attrs, dict):
                    mutated_attrs = mutate_attribute(attrs)
                else:
                    mutated_attrs = attrs
                new_props.append({"Attribute": mutated_attrs})
            else:
                new_props.append(prop)
        sv["properties"] = new_props
    return entity

def generate_uschema(i):
    model = copy.deepcopy(base_model)
    model["uSchemaModel"]["name"] = f"NoSQL-medical-Data-{i}"
    model["uSchemaModel"]["entities"] = [mutate_entity(ent) for ent in model["uSchemaModel"]["entities"]]
    return model

# Generate and save 10 new schema variations
for i in range(1, 11):
    uschema = generate_uschema(i)
    out_path = os.path.join(BASE_DIR, f"uschema_variant_{i}.json")
    with open(out_path, "w") as f:
        json.dump(uschema, f, indent=2)

print("✅ 10 U-Schema variants generated successfully!")
