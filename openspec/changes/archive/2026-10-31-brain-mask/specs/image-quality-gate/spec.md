# Spec Delta

## ADDED Requirements

### Requirement: Quality is measured before geometry and the mask fallback
Quality SHALL be measured on the loaded image before any geometry step, inside the brain mask as computed before its full-image fallback, with the same code at training and inference.

#### Scenario: Geometry does not change quality
- **WHEN** an image is prepared
- **THEN** the reported SNR and entropy equal those measured on the loaded image

#### Scenario: Fallback does not change quality
- **WHEN** an image whose brain mask is below the minimum fraction is measured
- **THEN** the SNR and entropy are computed on that mask, not on the full image

## REMOVED Requirements

### Requirement: Quality is measured on the original image
**Reason**: Its only scenario covers optional skull stripping, which the change `brain-mask` removes from preprocessing.
**Migration**: Replaced by "Quality is measured before geometry and the mask fallback", which keeps the rule and adds the mask fallback.
