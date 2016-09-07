using DataFrames, RDatasets
using Distributions
using DataFrames
input_data = open("LLCP2014.ASC","r") 


colNames = ["state","gender","race","ever_told_asthema","excercise_in_30","sleep_duration",
            "employee_status","income_level","use_tobacco","alcohol_con","smoked_least_100","freq_smoke",
            "stopped_smoked_12month","interval_last_smoke","still_asthema"]


function getValue(vals,stype)
    result = "NA"
    if stype == 1
        if vals != ' ' && vals != " " && vals != "  " && vals != '7' && vals !='9'
            result = string(vals)
        end
    elseif vals != ' ' && vals != " " && vals != "  " && vals != "77" && vals != "99"
            result = string(vals)
         end
    return result
end

function setAsthema()
    df = readtable("brfss_extracted_data.csv")

    print(df[1,5])
    df[:isAsthema] =  [df[x][5] == 1 ? "Yes" :"No" for x=1:size(df)[1]]
    print(size(df))
    delete!(df,:ever_told_asthema)
    delete!(df,:still_asthma)
    #for i in 1:12
    #names!(df,[symbol(colNames[i]) for i in 1:12])
    #end
    print(df)

    writetable("brfss_extracted_labeled_data.csv",df)
    
end

function extract_column_data(read_brfss)

    for line in readlines(read_brfss)
        line = chomp(line)
        state = getValue(line[1:2],1)                    #1
 
        gender = getValue(line[582],1)                   #2
        race_ethnicity = getValue(line[1419:1420],1)     #3
        ever_told_asthema = getValue(line[97],1)         #4
        exercise_in_30_days = getValue(line[91],1)       #5
        sleep_duration = getValue(line[92:93],2)         #6
        #println(".................. education level............")
        #println(line[1426])
        #education_level = getValue(line[1425],1)         #
        employment_status = getValue(line[151],1)        #7
        income_level = getValue(line[152:153],2)         #8
        use_of_tobacco_product = getValue(line[192],1)   #9 
        alcohol_cons = getValue(line[198:199],2)         #10
        smoked_least_100 = getValue(line[187],1)         #11
        freq_days_smoke = getValue(line[188],1)          #12
        stopped_smoke_12month = getValue(line[189],1)    #13
        interval_last_smoked = getValue(line[190:191],2) #14
        still_asthma_value = getValue(line[98],1)        #15                            
        write(output_file, "$state,$gender,$race_ethnicity,$ever_told_asthema,$exercise_in_30_days,$sleep_duration,$employment_status,$income_level,$use_of_tobacco_product,$alcohol_cons,$smoked_least_100,$freq_days_smoke,$stopped_smoke_12month,$interval_last_smoked,$still_asthma_value\n")
    end
    close(output_file)
    df = readtable("brfss_extracted_data.csv")
    names!(df,[:state,:gender,:race,:ever_told_asthema,:excercise_in_30,:sleep_duration,
               :employee_status,:income_level,:use_tobacco,:alcohol_con,:smoked_least_100,
               :freq_smoke,:stopped_smoked_12month,:interval_last_smoke,:still_asthma])
    writetable("brfss_extracted_data.csv",df)
end                          

function assign_value(categories, dist)

    idx = rand(dist, 1)

    return categories[idx]

end
    
function prepare_categories(countdict)

    sorted_keys = sort(collect(keys(countdict)))
    print("sorted key:")
    println(sorted_keys)
    res = Array{Int64, 1}(0)

    for k in sorted_keys

        if !isna(k)

            push!(res, k)

        end

    end
    print("res:")
    println(res)
    return res

end

function input_categories(v)
    n = length(v)

    res = similar(v)

    # get counts for each category

    count_dict = countmap(v)

    categories = prepare_categories(count_dict)

   

    n_obs = length(dropna(v))

    prob_vec = Array{Float64}(length(categories))

    # get vector of probabilities based on count data

    for (i, k) in enumerate(categories)

        prob_vec[i] = count_dict[k]/n_obs

    end

    D = Categorical(prob_vec)

    for i = 1:n
        res[i] = isna(v[i]) ? assign_value(categories, D)[1] : v[i]
    end

    return res
end    


function processNaValues(index)
    df = readtable("brfss_extracted_data.csv")
    for i in index
        res = input_categories(df[i])
        df[i]= res

    end

    #df = df[(df[:still_asthma] .== 0) | (df[:still_asthma] .== 1),:]

    deleterows!(df,find(isna(df[:,symbol("still_asthma")])))
    writetable("brfss_extracted_data.csv",df)
end

output_file_name = "brfss_extracted_data.csv" 
output_file = open(output_file_name,"w")  
    
extract_column_data(input_data)
#setAsthema()
close(output_file)

dd = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

processNaValues(dd)

