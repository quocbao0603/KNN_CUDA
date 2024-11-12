#include "arff_data.h"

ArffData::ArffData(): m_rel(""),
                      m_nominals(),
                      m_formats(),
                      m_num_attrs(0),
                      m_attrs(),
                      m_num_instances(0),
                      m_num_classes(0),
                      m_instances() {
}

ArffData::~ArffData() {
    {
        std::vector<ArffAttr*>::iterator itr;
        for(itr=m_attrs.begin();itr!=m_attrs.end();++itr) {
            delete *itr;
        }
    }
    {
        std::vector<ArffInstance*>::iterator itr;
        for(itr=m_instances.begin();itr!=m_instances.end();++itr) {
            delete *itr;
        }
    }
}

void ArffData::set_relation_name(const std::string& name) {
    m_rel = name;
}

std::string ArffData::get_relation_name() const {
    return m_rel;
}

int32 ArffData::num_attributes() const {
    return m_num_attrs;
}

int32 ArffData::num_classes() {
    if(m_num_classes != 0)
        return m_num_classes;
        
    int32 maxValue = 0;

    for(int i = 0; i < m_num_instances; i++)
    {
        int32 Class = m_instances[i]->get(m_num_attrs-1)->operator int32();

        if(Class > maxValue)
            maxValue = Class;
    }
    
    m_num_classes = maxValue+1;
    
    return m_num_classes;
}

void ArffData::add_attr(ArffAttr* attr) {
    m_attrs.push_back(attr);
    ++m_num_attrs;
}

ArffAttr* ArffData::get_attr(int32 idx) const {
    if((idx < 0) || (idx >= m_num_attrs)) {
        THROW("%s index out of bounds! idx=%d size=%d",
              "ArffData::get_attr", idx, m_num_attrs);
    }
    return m_attrs[idx];
}

int32 ArffData::num_instances() const {
    return m_num_instances;
}

void ArffData::add_instance(ArffInstance* inst) {
    _cross_check_instance(inst);
    m_instances.push_back(inst);
    ++m_num_instances;
}

ArffInstance* ArffData::get_instance(int32 idx) const {
    if((idx < 0) || (idx >= m_num_instances)) {
        THROW("%s index out of bounds! idx=%d size=%d",
              "ArffData::get_instance", idx, m_num_instances);
    }
    return m_instances[idx];
}

void ArffData::add_nominal_val(const std::string& name,
                               const std::string& val) {
    m_nominals[name].push_back(val);
}

ArffNominal ArffData::get_nominal(const std::string& name) {
    if(m_nominals.find(name) == m_nominals.end()) {
        THROW("ArffData::get_nominal list named '%s' does not exist!",
              name.c_str());
    }
    return m_nominals[name];
}

void ArffData::add_date_format(const std::string& name,
                               const std::string& val) {
    m_formats[name] = val;
}

std::string ArffData::get_date_format(const std::string& name) {
    if(m_formats.find(name) == m_formats.end()) {
        THROW("ArffData::get_date_format date named '%s' does not exist!",
              name.c_str());
    }
    return m_formats[name];
}

void ArffData::_cross_check_instance(ArffInstance* inst) {
    if(inst == NULL) {
        THROW("ArffData: input instance pointer is null!");
    }
    for(int32 i=0;i<m_num_attrs;++i) {
        ArffValue* val = inst->get(i);
        ArffAttr* attr = m_attrs[i];
        ArffValueEnum valType = val->type();
        ArffValueEnum attType = attr->type();
        bool a_is_num = (attr->type() == NUMERIC);
        bool a_is_nom = (attr->type() == NOMINAL);
        bool v_nan = ((valType != INTEGER) && (valType != FLOAT) && (valType != NUMERIC));
        bool v_nas = ((valType != STRING) && (valType != NOMINAL));
        // bad numeric/nominal
        if((a_is_num && v_nan) || (a_is_nom && v_nas)) {
            THROW("%s: attr-name=%s attr-type=%s, but inst-type=%s!",
                  "ArffData", attr->name().c_str(),
                  arff_value2str(attType).c_str(),
                  arff_value2str(valType).c_str());
        }
        // bad nominal value
        if(a_is_nom) {
            ArffNominal nom = get_nominal(attr->name());
            ArffNominal::iterator itr;
            std::string str = (std::string)*val;
            for(itr=nom.begin();itr!=nom.end();++itr) {
                if(str == *itr) {
                    break;
                }
            }
            if(itr == nom.end() && !val->missing()) {
                THROW("%s: attr:(name=%s type=%s) inst-val=%s not found!",
                      "ArffData", attr->name().c_str(),
                      arff_value2str(attType).c_str(), str.c_str());
            }
        }
        if(a_is_num || a_is_nom) {
            continue;
        }
        // data mismatch
        if(attType != valType) {
            THROW("%s: attr-name=%s attr-type=%s, but inst-type=%s!",
                  "ArffData", attr->name().c_str(),
                  arff_value2str(attType).c_str(),
                  arff_value2str(valType).c_str());
        }
    }
}

float* ArffData::get_dataset_matrix() {

    int num_instances = this->num_instances();
    int num_attributes = this->num_attributes();    
    
    float* matrix = (float*) malloc (num_instances * num_attributes * sizeof(float));

    for(int i = 0; i < num_instances; i++)
        for(int j = 0; j < num_attributes; j++)
            matrix[i*num_attributes + j] = this->get_instance(i)->get(j)->operator float();

    return matrix;
}