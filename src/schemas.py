from pydantic import BaseModel, field_validator, model_validator, FiniteFloat, Field, ConfigDict

class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    NPHI: list[FiniteFloat] = Field(description="Neutron porosity")
    RHOB: list[FiniteFloat]  = Field(description="Bulk density")
    GR: list[FiniteFloat]  = Field(description="Gamma ray")
    CALI: list[FiniteFloat]  = Field(description="Borehole diameter (Caliper)")
    PEF: list[FiniteFloat]  = Field(description="Photoelectric factor")
    DTC: list[FiniteFloat]  = Field(description="Sonic travel time")
    RDEP: list[FiniteFloat]  = Field(description="Deep resistivity")
    DEPTH_MD: list[FiniteFloat]  = Field(description="Measured depth")