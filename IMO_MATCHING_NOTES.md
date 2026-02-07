# IMO Matching Status and Limitations

## Summary

This document tracks the status of IMO (International Maritime Organization) number matching for IUU vessel identification.

## Current Status

### ✅ Successfully Mapped (50 vessels)
- **Source**: Extracted from `combined_iuu_list.csv` entries that contained both MMSI and IMO numbers
- **File**: `mmsi_imo_registry.csv`
- **Usage**: These pairs enable IMO-based IUU matching as a fallback when MMSI matching fails

### ❌ No IMO Available (13 vessels)
- **Source**: Entries in `combined_iuu_list.csv` that have MMSI but no IMO number
- **File**: `needs_imo_lookup.csv`
- **Status**: **Manually verified** - Searched Equasis and other maritime databases. No IMO numbers found for these vessels.

**Why no IMO?**
Many smaller fishing vessels, especially those involved in IUU activities, do not have IMO numbers assigned. IMO numbers are typically required for:
- Vessels above certain tonnage thresholds
- Vessels engaged in international shipping
- Commercial vessels meeting specific regulatory requirements

Smaller fishing vessels, particularly those operating illegally, often fall below these thresholds or operate outside regulatory frameworks, so they may not have IMO numbers recorded in public databases.

### Impact on Matching

**The system still works perfectly!**

1. **Primary matching**: Uses MMSI numbers (available for all 63 IUU vessels with MMSI)
2. **Fallback matching**: Uses IMO numbers for the 50 vessels that have them
3. **Coverage**: 
   - 50 vessels can be matched by both MMSI and IMO
   - 13 vessels can only be matched by MMSI (no IMO available)
   - Total: 63 IUU vessels with MMSI can be identified

## Files

- `mmsi_imo_registry.csv`: Contains 50 MMSI-IMO pairs (ready to use)
- `needs_imo_lookup.csv`: Contains 13 vessels without IMO numbers (documented as unavailable)
- `find_missing_imos.py`: Helper script to identify entries needing IMO lookups

## Notes

- The 13 vessels without IMO numbers will rely solely on MMSI matching, which is sufficient for identification
- IMO matching serves as an **enhancement** for vessels that have both identifiers, providing redundancy
- Future updates: If IMO numbers become available for any of these vessels, they can be added to `mmsi_imo_registry.csv`

## Verification Date

Manual verification completed: February 2026
- Checked Equasis database
- Cross-referenced multiple maritime databases
- Confirmed: No IMO numbers available for the 13 listed vessels
