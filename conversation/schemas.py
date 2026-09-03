from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class IntentOutput(BaseModel):
    label: str = Field(description="Classified intent: e.g. credential_request, financial_transfer, account_change, remote_access, normal_conversation, unknown")
    action: str = Field(description="Target action caller requests: e.g. share_otp, transfer_money, install_anydesk, disclose_aadhaar, none")

class InteractionSignals(BaseModel):
    urgency: bool = Field(description="True if caller conveys strict deadline, panic, or time limit")
    secrecy: bool = Field(description="True if caller tells victim not to inform bank, family, or manager")
    authority_pressure: bool = Field(description="True if caller impersonates police, CBI, RBI, bank manager, IT department")
    threat: bool = Field(default=False, description="True if caller threatens account block, arrest, electricity cutoff, legal penalty")
    credential_request: bool = Field(default=False, description="True if caller asks for OTP, PIN, CVV, password, or Aadhaar number")

class EmotionSignal(BaseModel):
    label: str = Field(description="Tone/emotion: e.g. neutral, excitement, urgency, aggression, panic, distress")
    confidence: float = Field(description="Confidence in emotion estimation (0.0 to 1.0)")

class ConversationOutput(BaseModel):
    transcript: str = Field(description="Full text transcript from ASR")
    intent: IntentOutput
    interaction: InteractionSignals
    emotion: EmotionSignal
    pressure: str = Field(description="Overall pressure level: 'low', 'medium', 'high'")

class SpeakerOutput(BaseModel):
    match_score: float = Field(description="Speaker consistency score (0.0 to 1.0)")
    confidence: float = Field(description="Confidence of speaker consistency evaluation")
    speaker_id: Optional[str] = Field(default=None, description="Enrolled reference profile ID evaluated")

class Person2Output(BaseModel):
    speaker: SpeakerOutput
    conversation: ConversationOutput
    timestamp: str
    latency_ms: float

    def to_person3_contract(self) -> Dict[str, Any]:
        """
        Returns the exact standardized JSON contract expected by Person 3 (Risk Engine).
        """
        return {
            "speaker_match": self.speaker.match_score,
            "transcript": self.conversation.transcript,
            "intent": self.conversation.intent.label,
            "interaction": {
                "urgency": self.conversation.interaction.urgency,
                "secrecy": self.conversation.interaction.secrecy,
                "authority_pressure": self.conversation.interaction.authority_pressure
            },
            "emotion": {
                "label": self.conversation.emotion.label,
                "confidence": self.conversation.emotion.confidence
            }
        }
