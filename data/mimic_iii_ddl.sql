-- =====================================================================
-- MIMIC-III Database Schema
-- Medical Information Mart for Intensive Care III
-- Version: 1.4
-- =====================================================================
-- 
-- This script creates the complete MIMIC-III database schema including:
-- - 26 core clinical tables
-- - Primary keys, foreign keys, and indexes
-- - Comments and documentation
--
-- Usage:
--   psql -U postgres -d mimiciii < mimic3_schema.sql
--
-- Note: This creates the structure only. Data loading is separate.
-- =====================================================================

-- Create schema and set search path
DROP SCHEMA IF EXISTS mimiciii CASCADE;
CREATE SCHEMA mimiciii;
SET search_path TO mimiciii, public;

-- =====================================================================
-- PATIENT INFORMATION TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- PATIENTS: Core patient demographics
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS patients CASCADE;
CREATE TABLE patients (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    gender VARCHAR(5) NOT NULL,
    dob TIMESTAMP NOT NULL,
    dod TIMESTAMP,
    dod_hosp TIMESTAMP,
    dod_ssn TIMESTAMP,
    expire_flag INT NOT NULL,
    CONSTRAINT patients_row_id_pk PRIMARY KEY (row_id),
    CONSTRAINT patients_subject_id_unique UNIQUE (subject_id)
);

COMMENT ON TABLE patients IS 'Patient demographics and death information';
COMMENT ON COLUMN patients.subject_id IS 'Unique patient identifier';
COMMENT ON COLUMN patients.gender IS 'Patient gender (M/F)';
COMMENT ON COLUMN patients.dob IS 'Date of birth';
COMMENT ON COLUMN patients.dod IS 'Date of death';
COMMENT ON COLUMN patients.expire_flag IS '1 if patient died, 0 otherwise';

-- ---------------------------------------------------------------------
-- ADMISSIONS: Hospital admission records
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS admissions CASCADE;
CREATE TABLE admissions (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    admittime TIMESTAMP NOT NULL,
    dischtime TIMESTAMP NOT NULL,
    deathtime TIMESTAMP,
    admission_type VARCHAR(50) NOT NULL,
    admission_location VARCHAR(50),
    discharge_location VARCHAR(50),
    insurance VARCHAR(255),
    language VARCHAR(10),
    religion VARCHAR(50),
    marital_status VARCHAR(50),
    ethnicity VARCHAR(200),
    edregtime TIMESTAMP,
    edouttime TIMESTAMP,
    diagnosis VARCHAR(255),
    hospital_expire_flag SMALLINT,
    has_chartevents_data SMALLINT NOT NULL,
    CONSTRAINT admissions_row_id_pk PRIMARY KEY (row_id),
    CONSTRAINT admissions_hadm_id_unique UNIQUE (hadm_id)
);

COMMENT ON TABLE admissions IS 'Hospital admission records for each patient';
COMMENT ON COLUMN admissions.hadm_id IS 'Unique hospital admission identifier';
COMMENT ON COLUMN admissions.admittime IS 'Admission timestamp';
COMMENT ON COLUMN admissions.dischtime IS 'Discharge timestamp';
COMMENT ON COLUMN admissions.hospital_expire_flag IS '1 if patient died during admission';

-- =====================================================================
-- ICU STAY TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- ICUSTAYS: ICU stay information
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS icustays CASCADE;
CREATE TABLE icustays (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    icustay_id INT NOT NULL,
    dbsource VARCHAR(20) NOT NULL,
    first_careunit VARCHAR(20) NOT NULL,
    last_careunit VARCHAR(20) NOT NULL,
    first_wardid SMALLINT NOT NULL,
    last_wardid SMALLINT NOT NULL,
    intime TIMESTAMP NOT NULL,
    outtime TIMESTAMP,
    los DOUBLE PRECISION,
    CONSTRAINT icustays_row_id_pk PRIMARY KEY (row_id),
    CONSTRAINT icustays_icustay_id_unique UNIQUE (icustay_id)
);

COMMENT ON TABLE icustays IS 'ICU stay information for patients';
COMMENT ON COLUMN icustays.icustay_id IS 'Unique ICU stay identifier';
COMMENT ON COLUMN icustays.intime IS 'ICU admission timestamp';
COMMENT ON COLUMN icustays.outtime IS 'ICU discharge timestamp';
COMMENT ON COLUMN icustays.los IS 'Length of stay in days';

-- ---------------------------------------------------------------------
-- TRANSFERS: Patient location transfers
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS transfers CASCADE;
CREATE TABLE transfers (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT,
    icustay_id INT,
    dbsource VARCHAR(20),
    eventtype VARCHAR(20),
    prev_careunit VARCHAR(20),
    curr_careunit VARCHAR(20),
    prev_wardid SMALLINT,
    curr_wardid SMALLINT,
    intime TIMESTAMP,
    outtime TIMESTAMP,
    los DOUBLE PRECISION,
    CONSTRAINT transfers_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE transfers IS 'Patient transfers between care units';
COMMENT ON COLUMN transfers.eventtype IS 'Type of transfer event';
COMMENT ON COLUMN transfers.curr_careunit IS 'Current care unit';

-- =====================================================================
-- CHARTED EVENTS TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- CHARTEVENTS: Charted observations and vital signs
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS chartevents CASCADE;
CREATE TABLE chartevents (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT,
    icustay_id INT,
    itemid INT,
    charttime TIMESTAMP,
    storetime TIMESTAMP,
    cgid INT,
    value VARCHAR(255),
    valuenum DOUBLE PRECISION,
    valueuom VARCHAR(50),
    warning INT,
    error INT,
    resultstatus VARCHAR(50),
    stopped VARCHAR(50),
    CONSTRAINT chartevents_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE chartevents IS 'Charted observations including vital signs';
COMMENT ON COLUMN chartevents.itemid IS 'Foreign key to D_ITEMS';
COMMENT ON COLUMN chartevents.charttime IS 'Time observation was charted';
COMMENT ON COLUMN chartevents.valuenum IS 'Numeric value of observation';

-- ---------------------------------------------------------------------
-- DATETIMEEVENTS: Date/time valued observations
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS datetimeevents CASCADE;
CREATE TABLE datetimeevents (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT,
    icustay_id INT,
    itemid INT NOT NULL,
    charttime TIMESTAMP NOT NULL,
    storetime TIMESTAMP NOT NULL,
    cgid INT NOT NULL,
    value TIMESTAMP,
    valueuom VARCHAR(50) NOT NULL,
    warning SMALLINT,
    error SMALLINT,
    resultstatus VARCHAR(50),
    stopped VARCHAR(50),
    CONSTRAINT datetimeevents_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE datetimeevents IS 'Date/time valued observations';

-- =====================================================================
-- INPUT/OUTPUT EVENTS TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- INPUTEVENTS_CV: Input events from CareVue system
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS inputevents_cv CASCADE;
CREATE TABLE inputevents_cv (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT,
    icustay_id INT,
    charttime TIMESTAMP,
    itemid INT,
    amount DOUBLE PRECISION,
    amountuom VARCHAR(30),
    rate DOUBLE PRECISION,
    rateuom VARCHAR(30),
    storetime TIMESTAMP,
    cgid INT,
    orderid INT,
    linkorderid INT,
    stopped VARCHAR(30),
    newbottle INT,
    originalamount DOUBLE PRECISION,
    originalamountuom VARCHAR(30),
    originalroute VARCHAR(30),
    originalrate DOUBLE PRECISION,
    originalrateuom VARCHAR(30),
    originalsite VARCHAR(30),
    CONSTRAINT inputevents_cv_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE inputevents_cv IS 'Input events from CareVue system (IV fluids, medications)';

-- ---------------------------------------------------------------------
-- INPUTEVENTS_MV: Input events from MetaVision system
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS inputevents_mv CASCADE;
CREATE TABLE inputevents_mv (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT,
    icustay_id INT,
    starttime TIMESTAMP,
    endtime TIMESTAMP,
    itemid INT,
    amount DOUBLE PRECISION,
    amountuom VARCHAR(30),
    rate DOUBLE PRECISION,
    rateuom VARCHAR(30),
    storetime TIMESTAMP,
    cgid INT,
    orderid INT,
    linkorderid INT,
    ordercategoryname VARCHAR(100),
    secondaryordercategoryname VARCHAR(100),
    ordercomponenttypedescription VARCHAR(200),
    ordercategorydescription VARCHAR(50),
    patientweight DOUBLE PRECISION,
    totalamount DOUBLE PRECISION,
    totalamountuom VARCHAR(50),
    isopenbag SMALLINT,
    continueinnextdept SMALLINT,
    cancelreason SMALLINT,
    statusdescription VARCHAR(30),
    comments_editedby VARCHAR(30),
    comments_canceledby VARCHAR(40),
    comments_date TIMESTAMP,
    originalamount DOUBLE PRECISION,
    originalrate DOUBLE PRECISION,
    CONSTRAINT inputevents_mv_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE inputevents_mv IS 'Input events from MetaVision system';

-- ---------------------------------------------------------------------
-- OUTPUTEVENTS: Output events (urine, drains, etc.)
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS outputevents CASCADE;
CREATE TABLE outputevents (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT,
    icustay_id INT,
    charttime TIMESTAMP,
    itemid INT,
    value DOUBLE PRECISION,
    valueuom VARCHAR(30),
    storetime TIMESTAMP,
    cgid INT,
    stopped VARCHAR(30),
    newbottle CHAR(1),
    iserror INT,
    CONSTRAINT outputevents_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE outputevents IS 'Output events (urine output, drains, etc.)';

-- =====================================================================
-- LABORATORY EVENTS TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- LABEVENTS: Laboratory test results
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS labevents CASCADE;
CREATE TABLE labevents (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT,
    itemid INT NOT NULL,
    charttime TIMESTAMP,
    value VARCHAR(200),
    valuenum DOUBLE PRECISION,
    valueuom VARCHAR(20),
    flag VARCHAR(20),
    CONSTRAINT labevents_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE labevents IS 'Laboratory test results';
COMMENT ON COLUMN labevents.flag IS 'Abnormal flag (abnormal/delta)';

-- ---------------------------------------------------------------------
-- MICROBIOLOGYEVENTS: Microbiology test results
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS microbiologyevents CASCADE;
CREATE TABLE microbiologyevents (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT,
    chartdate TIMESTAMP,
    charttime TIMESTAMP,
    spec_itemid INT,
    spec_type_desc VARCHAR(100),
    org_itemid INT,
    org_name VARCHAR(100),
    isolate_num SMALLINT,
    ab_itemid INT,
    ab_name VARCHAR(30),
    dilution_text VARCHAR(10),
    dilution_comparison VARCHAR(20),
    dilution_value DOUBLE PRECISION,
    interpretation VARCHAR(5),
    CONSTRAINT microbiologyevents_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE microbiologyevents IS 'Microbiology culture and sensitivity results';
COMMENT ON COLUMN microbiologyevents.org_name IS 'Organism name';
COMMENT ON COLUMN microbiologyevents.ab_name IS 'Antibiotic name';

-- =====================================================================
-- PROCEDURE EVENTS TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- PROCEDUREEVENTS_MV: Procedures from MetaVision
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS procedureevents_mv CASCADE;
CREATE TABLE procedureevents_mv (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    icustay_id INT,
    starttime TIMESTAMP,
    endtime TIMESTAMP,
    itemid INT,
    value DOUBLE PRECISION,
    valueuom VARCHAR(30),
    location VARCHAR(30),
    locationcategory VARCHAR(30),
    storetime TIMESTAMP,
    cgid INT,
    orderid INT,
    linkorderid INT,
    ordercategoryname VARCHAR(100),
    secondaryordercategoryname VARCHAR(100),
    ordercategorydescription VARCHAR(50),
    isopenbag SMALLINT,
    continueinnextdept SMALLINT,
    cancelreason SMALLINT,
    statusdescription VARCHAR(30),
    comments_editedby VARCHAR(30),
    comments_canceledby VARCHAR(30),
    comments_date TIMESTAMP,
    CONSTRAINT procedureevents_mv_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE procedureevents_mv IS 'Procedures from MetaVision system';

-- ---------------------------------------------------------------------
-- PROCEDURES_ICD: ICD-9 coded procedures
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS procedures_icd CASCADE;
CREATE TABLE procedures_icd (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    seq_num INT NOT NULL,
    icd9_code VARCHAR(10) NOT NULL,
    CONSTRAINT procedures_icd_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE procedures_icd IS 'ICD-9 coded procedures';
COMMENT ON COLUMN procedures_icd.icd9_code IS 'ICD-9 procedure code';

-- =====================================================================
-- DIAGNOSIS TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- DIAGNOSES_ICD: ICD-9 coded diagnoses
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS diagnoses_icd CASCADE;
CREATE TABLE diagnoses_icd (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    seq_num INT,
    icd9_code VARCHAR(10),
    CONSTRAINT diagnoses_icd_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE diagnoses_icd IS 'ICD-9 coded diagnoses';
COMMENT ON COLUMN diagnoses_icd.seq_num IS 'Sequence number (1=primary diagnosis)';

-- ---------------------------------------------------------------------
-- DRGCODES: Diagnosis Related Group codes
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS drgcodes CASCADE;
CREATE TABLE drgcodes (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    drg_type VARCHAR(20) NOT NULL,
    drg_code VARCHAR(20) NOT NULL,
    description VARCHAR(255),
    drg_severity SMALLINT,
    drg_mortality SMALLINT,
    CONSTRAINT drgcodes_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE drgcodes IS 'Diagnosis Related Group codes for billing';

-- =====================================================================
-- MEDICATION TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- PRESCRIPTIONS: Medication prescriptions
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS prescriptions CASCADE;
CREATE TABLE prescriptions (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    icustay_id INT,
    startdate TIMESTAMP,
    enddate TIMESTAMP,
    drug_type VARCHAR(100) NOT NULL,
    drug VARCHAR(100),
    drug_name_poe VARCHAR(100),
    drug_name_generic VARCHAR(100),
    formulary_drug_cd VARCHAR(120),
    gsn VARCHAR(200),
    ndc VARCHAR(120),
    prod_strength VARCHAR(120),
    dose_val_rx VARCHAR(120),
    dose_unit_rx VARCHAR(120),
    form_val_disp VARCHAR(120),
    form_unit_disp VARCHAR(120),
    route VARCHAR(120),
    CONSTRAINT prescriptions_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE prescriptions IS 'Medication prescriptions';
COMMENT ON COLUMN prescriptions.drug IS 'Drug name';
COMMENT ON COLUMN prescriptions.route IS 'Route of administration';

-- =====================================================================
-- CLINICAL NOTES TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- NOTEEVENTS: Clinical notes
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS noteevents CASCADE;
CREATE TABLE noteevents (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT,
    chartdate TIMESTAMP,
    charttime TIMESTAMP,
    storetime TIMESTAMP,
    category VARCHAR(50),
    description VARCHAR(255),
    cgid INT,
    iserror CHAR(1),
    text TEXT,
    CONSTRAINT noteevents_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE noteevents IS 'Clinical notes including discharge summaries, nursing notes, etc.';
COMMENT ON COLUMN noteevents.category IS 'Note category (e.g., Discharge summary, Nursing)';

-- =====================================================================
-- CAREGIVER TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- CAREGIVERS: Caregiver information
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS caregivers CASCADE;
CREATE TABLE caregivers (
    row_id INT NOT NULL,
    cgid INT NOT NULL,
    label VARCHAR(15),
    description VARCHAR(30),
    CONSTRAINT caregivers_row_id_pk PRIMARY KEY (row_id),
    CONSTRAINT caregivers_cgid_unique UNIQUE (cgid)
);

COMMENT ON TABLE caregivers IS 'Caregiver (healthcare provider) information';

-- =====================================================================
-- CPT EVENT TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- CPTEVENTS: CPT coded events
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS cptevents CASCADE;
CREATE TABLE cptevents (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    costcenter VARCHAR(10) NOT NULL,
    chartdate TIMESTAMP,
    cpt_cd VARCHAR(10) NOT NULL,
    cpt_number INT,
    cpt_suffix VARCHAR(5),
    ticket_id_seq INT,
    sectionheader VARCHAR(50),
    subsectionheader VARCHAR(255),
    description VARCHAR(200),
    CONSTRAINT cptevents_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE cptevents IS 'CPT (Current Procedural Terminology) coded events';

-- =====================================================================
-- SERVICES TABLES
-- =====================================================================

-- ---------------------------------------------------------------------
-- SERVICES: Hospital services
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS services CASCADE;
CREATE TABLE services (
    row_id INT NOT NULL,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    transfertime TIMESTAMP NOT NULL,
    prev_service VARCHAR(20),
    curr_service VARCHAR(20),
    CONSTRAINT services_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE services IS 'Hospital service assignments for patients';

-- =====================================================================
-- DEFINITION TABLES (DICTIONARIES)
-- =====================================================================

-- ---------------------------------------------------------------------
-- D_ITEMS: Item definitions
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS d_items CASCADE;
CREATE TABLE d_items (
    row_id INT NOT NULL,
    itemid INT NOT NULL,
    label VARCHAR(200),
    abbreviation VARCHAR(100),
    dbsource VARCHAR(20),
    linksto VARCHAR(50),
    category VARCHAR(100),
    unitname VARCHAR(100),
    param_type VARCHAR(30),
    conceptid INT,
    CONSTRAINT d_items_row_id_pk PRIMARY KEY (row_id),
    CONSTRAINT d_items_itemid_unique UNIQUE (itemid)
);

COMMENT ON TABLE d_items IS 'Dictionary of charted items';
COMMENT ON COLUMN d_items.label IS 'Item description';
COMMENT ON COLUMN d_items.linksto IS 'Table this item links to';

-- ---------------------------------------------------------------------
-- D_LABITEMS: Laboratory item definitions
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS d_labitems CASCADE;
CREATE TABLE d_labitems (
    row_id INT NOT NULL,
    itemid INT NOT NULL,
    label VARCHAR(100) NOT NULL,
    fluid VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    loinc_code VARCHAR(100),
    CONSTRAINT d_labitems_row_id_pk PRIMARY KEY (row_id),
    CONSTRAINT d_labitems_itemid_unique UNIQUE (itemid)
);

COMMENT ON TABLE d_labitems IS 'Dictionary of laboratory test items';
COMMENT ON COLUMN d_labitems.loinc_code IS 'LOINC code for test';

DROP TABLE IF EXISTS d_icd_diagnoses CASCADE;
CREATE TABLE d_icd_diagnoses (
    row_id INT NOT NULL,
    icd9_code VARCHAR(10) NOT NULL,
    short_title VARCHAR(50) NOT NULL,
    long_title VARCHAR(255) NOT NULL,
    CONSTRAINT d_icd_diagnoses_row_id_pk PRIMARY KEY (row_id),
    CONSTRAINT d_icd_diagnoses_icd9_unique UNIQUE (icd9_code)
);

COMMENT ON TABLE d_icd_diagnoses IS 'Dictionary of ICD-9 diagnosis codes';

DROP TABLE IF EXISTS d_icd_procedures CASCADE;
CREATE TABLE d_icd_procedures (
    row_id INT NOT NULL,
    icd9_code VARCHAR(10) NOT NULL,
    short_title VARCHAR(50) NOT NULL,
    long_title VARCHAR(255) NOT NULL,
    CONSTRAINT d_icd_procedures_row_id_pk PRIMARY KEY (row_id),
    CONSTRAINT d_icd_procedures_icd9_unique UNIQUE (icd9_code)
);

COMMENT ON TABLE d_icd_procedures IS 'Dictionary of ICD-9 procedure codes';

DROP TABLE IF EXISTS d_cpt CASCADE;
CREATE TABLE d_cpt (
    row_id INT NOT NULL,
    category SMALLINT NOT NULL,
    sectionrange VARCHAR(100) NOT NULL,
    sectionheader VARCHAR(50) NOT NULL,
    subsectionrange VARCHAR(100) NOT NULL,
    subsectionheader VARCHAR(255) NOT NULL,
    codesuffix VARCHAR(5),
    mincodeinsubsection INT NOT NULL,
    maxcodeinsubsection INT NOT NULL,
    CONSTRAINT d_cpt_row_id_pk PRIMARY KEY (row_id)
);

COMMENT ON TABLE d_cpt IS 'Dictionary of CPT codes';

ALTER TABLE admissions
    ADD CONSTRAINT admissions_subject_id_fk
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id);

ALTER TABLE icustays
    ADD CONSTRAINT icustays_subject_id_fk
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id);

ALTER TABLE icustays
    ADD CONSTRAINT icustays_hadm_id_fk
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id);

ALTER TABLE transfers
    ADD CONSTRAINT transfers_subject_id_fk
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id);

ALTER TABLE transfers
    ADD CONSTRAINT transfers_hadm_id_fk
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id);

ALTER TABLE transfers
    ADD CONSTRAINT transfers_icustay_id_fk
    FOREIGN KEY (icustay_id) REFERENCES icustays(icustay_id);

ALTER TABLE chartevents
    ADD CONSTRAINT chartevents_subject_id_fk
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id);

ALTER TABLE chartevents
    ADD CONSTRAINT chartevents_hadm_id_fk
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id);

ALTER TABLE chartevents
    ADD CONSTRAINT chartevents_icustay_id_fk
    FOREIGN KEY (icustay_id) REFERENCES icustays(icustay_id);

ALTER TABLE chartevents
    ADD CONSTRAINT chartevents_itemid_fk
    FOREIGN KEY (itemid) REFERENCES d_items(itemid);

ALTER TABLE chartevents
    ADD CONSTRAINT chartevents_cgid_fk
    FOREIGN KEY (cgid) REFERENCES caregivers(cgid);

ALTER TABLE labevents
    ADD CONSTRAINT labevents_subject_id_fk
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id);

ALTER TABLE labevents
    ADD CONSTRAINT labevents_hadm_id_fk
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id);

ALTER TABLE labevents
    ADD CONSTRAINT labevents_itemid_fk
    FOREIGN KEY (itemid) REFERENCES d_labitems(itemid);

ALTER TABLE diagnoses_icd
    ADD CONSTRAINT diagnoses_icd_subject_id_fk
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id);

ALTER TABLE diagnoses_icd
    ADD CONSTRAINT diagnoses_icd_hadm_id_fk
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id);

ALTER TABLE diagnoses_icd
    ADD CONSTRAINT diagnoses_icd_icd9_fk
    FOREIGN KEY (icd9_code) REFERENCES d_icd_diagnoses(icd9_code);

ALTER TABLE procedures_icd
    ADD CONSTRAINT procedures_icd_subject_id_fk
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id);

ALTER TABLE procedures_icd
    ADD CONSTRAINT procedures_icd_hadm_id_fk
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id);

ALTER TABLE procedures_icd
    ADD CONSTRAINT procedures_icd_icd9_fk
    FOREIGN KEY (icd9_code) REFERENCES d_icd_procedures(icd9_code);

ALTER TABLE prescriptions
    ADD CONSTRAINT prescriptions_subject_id_fk
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id);

ALTER TABLE prescriptions
    ADD CONSTRAINT prescriptions_hadm_id_fk
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id);

ALTER TABLE prescriptions
    ADD CONSTRAINT prescriptions_icustay_id_fk
    FOREIGN KEY (icustay_id) REFERENCES icustays(icustay_id);

ALTER TABLE noteevents
    ADD CONSTRAINT noteevents_subject_id_fk
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id);

ALTER TABLE noteevents
    ADD CONSTRAINT noteevents_hadm_id_fk
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id);

ALTER TABLE noteevents
    ADD CONSTRAINT noteevents_cgid_fk
    FOREIGN KEY (cgid) REFERENCES caregivers(cgid);

CREATE INDEX patients_subject_id_idx ON patients(subject_id);

CREATE INDEX admissions_subject_id_idx ON admissions(subject_id);
CREATE INDEX admissions_hadm_id_idx ON admissions(hadm_id);
CREATE INDEX admissions_admittime_idx ON admissions(admittime);

CREATE INDEX icustays_subject_id_idx ON icustays(subject_id);
CREATE INDEX icustays_hadm_id_idx ON icustays(hadm_id);
CREATE INDEX icustays_icustay_id_idx ON icustays(icustay_id);
CREATE INDEX icustays_intime_idx ON icustays(intime);

CREATE INDEX chartevents_subject_id_idx ON chartevents(subject_id);
CREATE INDEX chartevents_hadm_id_idx ON chartevents(hadm_id);
CREATE INDEX chartevents_icustay_id_idx ON chartevents(icustay_id);
CREATE INDEX chartevents_itemid_idx ON chartevents(itemid);
CREATE INDEX chartevents_charttime_idx ON chartevents(charttime);

CREATE INDEX labevents_subject_id_idx ON labevents(subject_id);
CREATE INDEX labevents_hadm_id_idx ON labevents(hadm_id);
CREATE INDEX labevents_itemid_idx ON labevents(itemid);
CREATE INDEX labevents_charttime_idx ON labevents(charttime);

CREATE INDEX diagnoses_icd_subject_id_idx ON diagnoses_icd(subject_id);
CREATE INDEX diagnoses_icd_hadm_id_idx ON diagnoses_icd(hadm_id);
CREATE INDEX diagnoses_icd_icd9_code_idx ON diagnoses_icd(icd9_code);

CREATE INDEX procedures_icd_subject_id_idx ON procedures_icd(subject_id);
CREATE INDEX procedures_icd_hadm_id_idx ON procedures_icd(hadm_id);
CREATE INDEX procedures_icd_icd9_code_idx ON procedures_icd(icd9_code);

CREATE INDEX prescriptions_subject_id_idx ON prescriptions(subject_id);
CREATE INDEX prescriptions_hadm_id_idx ON prescriptions(hadm_id);
CREATE INDEX prescriptions_icustay_id_idx ON prescriptions(icustay_id);

CREATE INDEX noteevents_subject_id_idx ON noteevents(subject_id);
CREATE INDEX noteevents_hadm_id_idx ON noteevents(hadm_id);
CREATE INDEX noteevents_charttime_idx ON noteevents(charttime);

CREATE INDEX microbiologyevents_subject_id_idx ON microbiologyevents(subject_id);
CREATE INDEX microbiologyevents_hadm_id_idx ON microbiologyevents(hadm_id);
CREATE INDEX microbiologyevents_charttime_idx ON microbiologyevents(charttime);

CREATE INDEX inputevents_cv_subject_id_idx ON inputevents_cv(subject_id);
CREATE INDEX inputevents_cv_hadm_id_idx ON inputevents_cv(hadm_id);
CREATE INDEX inputevents_cv_icustay_id_idx ON inputevents_cv(icustay_id);
CREATE INDEX inputevents_cv_charttime_idx ON inputevents_cv(charttime);

CREATE INDEX inputevents_mv_subject_id_idx ON inputevents_mv(subject_id);
CREATE INDEX inputevents_mv_hadm_id_idx ON inputevents_mv(hadm_id);
CREATE INDEX inputevents_mv_icustay_id_idx ON inputevents_mv(icustay_id);
CREATE INDEX inputevents_mv_starttime_idx ON inputevents_mv(starttime);

CREATE INDEX outputevents_subject_id_idx ON outputevents(subject_id);
CREATE INDEX outputevents_hadm_id_idx ON outputevents(hadm_id);
CREATE INDEX outputevents_icustay_id_idx ON outputevents(icustay_id);
CREATE INDEX outputevents_charttime_idx ON outputevents(charttime);

