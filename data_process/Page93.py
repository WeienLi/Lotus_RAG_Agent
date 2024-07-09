import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def update_page_content(data, page_number, new_content):
    for entry in data:
        if entry.get("page_number") == page_number:
            entry["content"] = new_content
            break

if __name__ == "__main__":
    input_file_path = "./processed_best.json"
    page_number_to_update = 93
    new_text_content = """1969\nType 50\nElan +2S\n\nStill officially referred to in official Lotus literature as the Lotus +2 rather than Elan +2, the +2S model launched in early 1969 was notable as the first ever Lotus that was not sold in kit form, only being sold in fully-built guise. Some felt that this change was due to the new Government taxation systems which would eventually do away with the Purchase Tax and bring in the new VAT tax, as well as the desire of Colin Chapman for Lotus to move up-market. He felt that offering "Kit" cars was not the image Lotus needed to move forward into a new era. Thus, in an attempt to push the car up-market (more luxurious and more appealing) all +2S cars would be factory built and assembled. This is probably also why the Elan part of the name was dropped from marketing, to help distance it from the component form Elan (in the UK market). Interestingly though, many cars that appeared to retain the "Elan +2" badge.\n\nIn the words of the Lotus Cars brochure produced for the launch, "The Lotus +2S is for the discerning motorist who requires the renowned Lotus road-holding and performance coupled to an attractive two plus two body. The +S combines the hand-built reliability, safety and comfort of previous Lotus models engineered to an even higher degree of luxury, a luxury which offers forty extra refinements to the previous mechanical and coachwork specification list."\n\nThe new +2S model offered only as the fully built up and all-inclusive price of £2375 was fitted with the 118bhp "Special Equipment" version of the Lotus Twin-cam engine (Lotus quoted figures of over 20,000 units produced by 1969) mated to a close-ratio version of the standard Ford 4-speed gearbox. The +2S also included revised seats and centre console, a new dashboard layout which placed the rocker switches on a mahogany facia and auxiliary fog lights at the front end, set below the bumper on either side of the front grill. Proudly advertised as "The car with no extras," and "so few extras that it not worth printing a list," it is somewhat strange to note that factory options available included the following: Aluminium alloy wheels (Brand Lotus), tinted windows (either just the front windscreen, or all round), heated rear screen, rear seat belts and most importantly to some, a workshop manual. Previously the manual had only been available as a cheaply photocopied typewritten supplement to the workshop manual for a standard model Elan.\n\nWith this conscious decision to move up-market, the +2S also offered several practical enhancements to the comfort and reliability aspects of the new model rather than the minor cosmetic/mechanical aspects of any other sports car. The launch brochures were keen to push away accidents when in capable hands."\n\nThe Lotus +2 had certainly moved the company into a new market sector, and the +2S took things a step further. Sales figures were good and the car opened Lotus up to a new clientele that would stay with the company for many years to come.\n\nMODEL  Type 50\nNAME/FORMULA  Lotus (Elan) +2S\nYEARS OF PRODUCTION  1969-71\nEXAMPLES BUILT  3576\nENGINE TYPE  Lotus-Ford Twin-cam\nENGINE SIZE/POWER  1588cc/118bhp\nLENGTH/WIDTH/HEIGHT  168in/64in/48in\nWHEELBASE  96in\nWEIGHT  1960-1980lb/884-898kg\n"""

    # Load JSON database
    data = load_json(input_file_path)
    print("JSON database loaded.")

    # Update the specific page content
    update_page_content(data, page_number_to_update, new_text_content)
    print(f"Content for page {page_number_to_update} updated.")

    # Save the updated JSON back to the file
    save_json(data, input_file_path)
    print("JSON database saved.")
