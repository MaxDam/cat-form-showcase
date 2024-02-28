from pydantic import BaseModel, Field
from cat.mad_hatter.decorators import plugin
    
class MySettings(BaseModel):
    strict: bool = Field(
        title="strict",
        default=False
    )
    ask_confirm: bool = Field(
        title="ask confirm",
        default=True
    )
    
@plugin
def settings_schema():
    return MySettings.schema()
