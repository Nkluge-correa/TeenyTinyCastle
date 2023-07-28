# The Basic PE Data Structure

Return to the [castle](https://github.com/Nkluge-correa/teeny-tiny_castle).

## DOS Header

_Every PE file starts with a 64-byte-long structure called the DOS header. It's what makes the PE file an MS-DOS executable. This header possesses two main features:_

- `e_magic` → - a constant numerical or text value used to identify a file format or protocol (e.g., 'MZ' (0x5a4d) refers to `Mark Zbikowski` the designer of MS-DOS executable file format.). Go [here](https://en.wikipedia.org/wiki/List_of_file_signatures "List of file signatures") for a list of file signatures.
- `e_lfnew` → a pointer to the NT header.
- `DOS Stub` → most Windows programs' DOS header contains a DOS program which does nothing but print "_This program cannot be run in DOS mode_".

## NT Headers

_The NT Headers part contains three main parts:_

- `PE signature` → A 4-byte signature that identifies the file as a PE file (0x5045 = 'PE' in ASCII).
- `File Header` → A standard `COFF` File Header. It holds some information about the PE file:
- `Machine` → the architecture this binary is supposed to run on (`0x014C` == x86 binary and `0x8664` == x86-x64 binary).
- `TimeDateStamp` → UNIX timestamp (seconds since epoch or 00:00:00 1/1/1970).
- `NumberOfSections` → number of section headers.
- `Characteristics` → specify some characteristics of the PE file.

## Optional Header

_The most important header of the NT Headers, its name is the Optional Header because some files, like object files, don't have it. This header provides important information to the OS loader:_

- `Magic` → depending on this value, the binary will be interpreted as a 32-bit or 64-bit binary (`0x10B` == 32-bit and `0x20B` == 64-bit).
- `AddressOfEntryPoint` → specifies the RVA (relative virtual address).
- `ImageBase` → specifies the preferred virtual memory location where the beginning of the binary should be placed.
- `SectionAlignment` → specifies that sections must be aligned on boundaries that are multiples of this value.
- `FileAlignment` → if the data was written to the binary into chunks no smaller than this value.
- `SizeOfImage` → the amount of contiguous memory that must be reserved to load the binary into memory.
- `DllCharacteristics` → Specify some security characteristics for the PE file.
- `DataDirectory` → an array of Data Entries.

## Sections

_Sections are where the actual contents of the file are stored, these include things like data and resources that the program uses, and also the actual code of the program. Some common section names are:_

- `.text` → the actual code the binary runs.
- `.data` → read/write data (globals).
- `.rdata` → read-only data (strings).
- `.bss` → Block Storage Segment (uninitialized data format), often merged with the .data section.
- `.idata` → import address table, often merged with .text or .rdata sections.
- `.edata` → export address table.
- `.pdata` → some architectures like ARM, and MIPS use these section structures to aid stack-walking at run-time.
- `PAGE` → code/data which it's fine to page out to disk if you're running out of memory.
- `.reolc` → relocation information for where to modify the hardcoded addresses.
- `.rsrc` → resources like icons and other embedded binaries, this section has a structure organizing it like a filesystem.

_Some common structure of a section header is:_

- `Name`.
- `VirtualSize`.
- `VirtualAddress`.
- `SizeOfRawData`.
- `PointerToRawData`.
- `Characteristics`.

---

Return to the [castle](https://github.com/Nkluge-correa/teeny-tiny_castle).
