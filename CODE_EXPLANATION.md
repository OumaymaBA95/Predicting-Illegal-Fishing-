# Code Explanation: Fishing Vessel Data Processing

## What This Code Does (Simple Overview)

This code processes **massive amounts of fishing vessel tracking data** to identify which boats are fishing in two specific regions (Gulf of Mexico and Mediterranean Sea), and then labels which ones are known illegal vessels. Think of it like a detective sorting through millions of GPS records to find suspicious fishing activity.

---

## Step-by-Step Breakdown

### **Step 1: Load the "Bad Guy" List** (Lines 25-34)

**What it does:**
- Reads a CSV file (`IUUList-20251108.csv`) that contains a list of **known illegal fishing vessels**
- Each vessel has a unique ID called an **MMSI** (like a license plate number for boats)
- Creates a "cheat sheet" (dictionary) that says: "If you see MMSI #12345, that's an illegal vessel"

**Why it matters:**
- This is your "ground truth" - you know these boats are bad, so you can use them to train your machine learning model later
- The dictionary lookup is super fast (like looking up a word in a dictionary vs. reading the whole book)

**Example:**
```
MMSI: "123456789" → is_iuu: 1 (illegal)
MMSI: "987654321" → is_iuu: 0 (not on the list)
```

---

### **Step 2: Find All the Data Files** (Lines 36-40)

**What it does:**
- Looks in a specific folder for all CSV files from 2024
- These files contain daily fishing activity data from Global Fishing Watch
- Each file has millions of rows tracking where boats were and how long they fished

**Why it matters:**
- You have 365 days of data (one file per day), so you need to process them all
- The code automatically finds all files matching the pattern

---

### **Step 3: Process Each File in Small Chunks** (Lines 42-57)

**What it does:**
- Reads each massive CSV file in **chunks of 100,000 rows** (instead of loading everything at once)
- For each chunk:
  1. **Filters** to only keep boats in two regions:
     - **Gulf of Mexico**: latitude 20-30°, longitude -98° to -80°
     - **Mediterranean Sea**: latitude 30-46°, longitude -6° to 36°
  2. **Groups by vessel** (MMSI) and calculates:
     - Total fishing hours (sum)
     - Average latitude (mean)
     - Average longitude (mean)
  3. Saves these summaries to a list

**Why it matters:**
- Files are **HUGE** (could be gigabytes). Reading in chunks prevents your computer from running out of memory
- You only care about two specific regions, so filtering early saves processing time
- Grouping by vessel gives you one row per boat instead of thousands

**Example transformation:**
```
Before: 10,000 rows for vessel #12345 (one row per day/location)
After:  1 row for vessel #12345 (total hours: 450, avg lat: 25.5, avg lon: -90.2)
```

---

### **Step 4: Combine Everything** (Lines 59-63)

**What it does:**
- Takes all the summaries from all files and combines them
- Since the same boat might appear in multiple files, it **re-aggregates**:
  - Sums up fishing hours across all days
  - Recalculates average location across all days
- Creates one final dataset with one row per vessel

**Why it matters:**
- A boat might fish for 5 hours on Monday and 8 hours on Tuesday
- After this step, you have: 13 total hours, and one average location

---

### **Step 5: Label Illegal Vessels** (Lines 65-69)

**What it does:**
- For each vessel in your final dataset, checks if it's in the "bad guy" list
- Adds a new column `is_iuu`:
  - `1` = known illegal vessel
  - `0` = not on the illegal list (could still be suspicious, but not confirmed)

**Why it matters:**
- This is the **label** you'll use for machine learning
- Your model will learn: "Boats with these patterns tend to be illegal"

**Final Output:**
```
MMSI        | fishing_hours | avg_lat | avg_lon | is_iuu
123456789   | 450          | 25.5    | -90.2   | 1 (illegal!)
987654321   | 320          | 35.2    | 12.8    | 0 (not on list)
```

---

## Key Concepts Explained Simply

### **MMSI (Maritime Mobile Service Identity)**
- Like a social security number for boats
- Every vessel that uses AIS (Automatic Identification System) has one
- Used to track the same boat across different days/files

### **Chunking**
- Instead of loading a 5GB file all at once, read it in 100,000-row pieces
- Like reading a book one chapter at a time instead of trying to hold the whole book

### **Aggregation**
- Taking many detailed records and summarizing them
- Like: "Instead of listing every time you went to the store, just tell me how many times total"

### **Dictionary Lookup**
- Super fast way to check if something exists
- Like: "Is MMSI 12345 in the bad list?" → Dictionary says "Yes, it's illegal" instantly

---

## What This Code Produces

At the end, you have a **clean dataset** with:
- One row per fishing vessel
- Total fishing hours in the target regions
- Average location (latitude/longitude)
- Label indicating if it's a known illegal vessel

This dataset is ready to be used for **machine learning** to predict which other vessels might be illegal based on their behavior patterns.

---

## Why This Approach is Efficient

1. **Memory-friendly**: Processes huge files without crashing
2. **Fast filtering**: Only processes data from regions you care about
3. **Quick lookups**: Dictionary is much faster than searching through lists
4. **Scalable**: Can handle years of data, not just 2024

---

## Presentation Tips

When presenting this code, you could say:

> "This code processes millions of GPS tracking records from fishing vessels to identify activity in two critical regions: the Gulf of Mexico and the Mediterranean Sea. It efficiently handles massive datasets by processing them in manageable chunks, then summarizes each vessel's total fishing activity and location. Most importantly, it labels which vessels are known illegal operators, creating a labeled dataset that will train our machine learning model to identify suspicious fishing patterns." 
